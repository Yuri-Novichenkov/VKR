import argparse
import os
import sys
from pathlib import Path

import mlflow
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LiDARDataset
from src.models import (
    PointNetSegmentation,
    PointNetPlusPlusSegmentation,
    DGCNNSegmentation,
    DGCNNClassification,
    LDGCNNSegmentation,
    LDGCNNClassification,
)


def calculate_metrics(predictions, targets, num_classes, task="segmentation"):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    if task == "classification":
        accuracy = accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        report = classification_report(targets, predictions, labels=list(range(num_classes)), output_dict=True, zero_division=0)
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report,
        }

    accuracy = accuracy_score(targets.flatten(), predictions.flatten())
    cm = confusion_matrix(targets.flatten(), predictions.flatten(), labels=list(range(num_classes)))
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(intersection / union if union > 0 else 0.0)
    mean_iou = float(sum(ious) / len(ious))
    report = classification_report(
        targets.flatten(),
        predictions.flatten(),
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": ious,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def build_model(model_type, task, num_classes, num_features):
    if model_type == "pointnet":
        return PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "pointnet++":
        return PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "dgcnn":
        return DGCNNSegmentation(num_classes=num_classes, num_features=num_features) if task == "segmentation" else DGCNNClassification(num_classes=num_classes, num_features=num_features)
    if model_type == "ldgcnn":
        return LDGCNNSegmentation(num_classes=num_classes, num_features=num_features) if task == "segmentation" else LDGCNNClassification(num_classes=num_classes, num_features=num_features)
    raise ValueError(f"Неизвестная модель: {model_type}")


def resolve_test_path(args):
    if args.test_data:
        return args.test_data
    data_root = args.data_root or os.path.join("Files", args.dataset, "LiDAR")
    return os.path.join(data_root, f"{args.dataset}_test.txt")


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели")
    parser.add_argument("--test_data", type=str, default=None, help="Путь к тестовому набору данных")
    parser.add_argument("--data_root", type=str, default=None, help="Корневая папка с данными (Files/Mar18/LiDAR)")
    parser.add_argument("--dataset", type=str, default="Mar16", help="Префикс датасета")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--num_points", type=int, default=4096, help="Количество точек в облаке")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--save_results", type=str, default=None, help="Путь для сохранения результатов")
    parser.add_argument("--experiment_name", type=str, default="PointCloudExperiments", help="MLflow experiment name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Использование устройства: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    num_features = checkpoint["num_features"]
    model_type = checkpoint.get("model_type", "pointnet")
    task = checkpoint.get("task", "segmentation")

    model = build_model(model_type, task, num_classes, num_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_path = resolve_test_path(args)
    print(f"Тестовые данные: {test_path}")

    test_dataset = LiDARDataset(test_path, num_points=args.num_points, augment=False, has_labels=True, task=task)
    test_dataset.num_classes = num_classes
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            if task == "classification":
                predictions = model(features)
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                if model_type == "pointnet":
                    predictions, _, _ = model(features)
                else:
                    predictions = model(features)
                pred_classes = torch.argmax(predictions, dim=2)

            all_predictions.append(pred_classes)
            all_targets.append(labels)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_predictions, all_targets, num_classes, task=task)

    print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if task == "segmentation":
        print(f"mIoU: {metrics['mean_iou']:.4f}")

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n")
            f.write("=" * 50 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            if task == "segmentation":
                f.write(f"mIoU: {metrics['mean_iou']:.4f}\n")

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{model_type}_{task}_test"):
        mlflow.log_params({"model": model_type, "task": task, "dataset": args.dataset})
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        if task == "segmentation":
            mlflow.log_metric("test_miou", metrics["mean_iou"])


if __name__ == "__main__":
    main()
