import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LiDARDataset
from src.models import (
    PointNetSegmentation,
    PointNetPlusPlusSegmentation,
    DGCNNSegmentation,
    LDGCNNSegmentation,
)


def build_model(model_type, num_classes, num_features):
    if model_type == "pointnet":
        return PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "pointnet++":
        return PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "dgcnn":
        return DGCNNSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "ldgcnn":
        return LDGCNNSegmentation(num_classes=num_classes, num_features=num_features)
    raise ValueError(f"Неизвестный тип модели: {model_type}")


def calculate_metrics(predictions, targets, num_classes):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    accuracy = (predictions.flatten() == targets.flatten()).mean()

    cm = pd.crosstab(
        targets.flatten(),
        predictions.flatten(),
        rownames=["true"],
        colnames=["pred"],
        dropna=False,
    ).reindex(index=range(num_classes), columns=range(num_classes), fill_value=0).values

    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(intersection / union if union > 0 else 0.0)
    mean_iou = float(sum(ious) / len(ious))

    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": ious,
        "confusion_matrix": cm,
    }


def evaluate_model(model, model_type, loader, device, num_classes):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.float().to(device)
            labels = labels.long().to(device)
            if model_type == "pointnet":
                predictions, _, _ = model(features)
            else:
                predictions = model(features)
            pred_classes = torch.argmax(predictions, dim=2)
            all_predictions.append(pred_classes)
            all_targets.append(labels)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    return metrics


def compare_models(model_a_checkpoint, model_b_checkpoint, test_data, num_points=4096, batch_size=8, save_dir="comparison"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_a = torch.load(model_a_checkpoint, map_location=device, weights_only=False)
    checkpoint_b = torch.load(model_b_checkpoint, map_location=device, weights_only=False)

    model_a_type = checkpoint_a.get("model_type", "pointnet")
    model_b_type = checkpoint_b.get("model_type", "pointnet++")

    num_classes = checkpoint_a["num_classes"]
    num_features = checkpoint_a["num_features"]

    model_a = build_model(model_a_type, num_classes, num_features).to(device)
    model_b = build_model(model_b_type, num_classes, num_features).to(device)
    model_a.load_state_dict(checkpoint_a["model_state_dict"])
    model_b.load_state_dict(checkpoint_b["model_state_dict"])

    test_dataset = LiDARDataset(test_data, num_points=num_points, augment=False, has_labels=True, task="segmentation")
    test_dataset.num_classes = num_classes

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    metrics_a = evaluate_model(model_a, model_a_type, test_loader, device, num_classes)
    metrics_b = evaluate_model(model_b, model_b_type, test_loader, device, num_classes)

    df = pd.DataFrame(
        {
            "Метрика": ["Loss", "Accuracy", "mIoU"],
            model_a_type: [metrics_a.get("loss", 0), metrics_a["accuracy"], metrics_a["mean_iou"]],
            model_b_type: [metrics_b.get("loss", 0), metrics_b["accuracy"], metrics_b["mean_iou"]],
        }
    )

    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ["Loss", "Accuracy", "mIoU"]
    a_values = [metrics_a.get("loss", 0), metrics_a["accuracy"], metrics_a["mean_iou"]]
    b_values = [metrics_b.get("loss", 0), metrics_b["accuracy"], metrics_b["mean_iou"]]

    for i, (ax, metric_name, a_val, b_val) in enumerate(zip(axes, metrics_names, a_values, b_values)):
        ax.bar(i - 0.2, a_val, 0.4, label=model_a_type, alpha=0.8)
        ax.bar(i + 0.2, b_val, 0.4, label=model_b_type, alpha=0.8)
        ax.set_title(metric_name)
        ax.set_xticks([i])
        ax.set_xticklabels([metric_name])
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

    results_path = os.path.join(save_dir, "comparison_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(df.to_string(index=False))

    print(f"Результаты сохранены: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Сравнение моделей")
    parser.add_argument("--model_a_checkpoint", type=str, required=True, help="Чекпоинт модели A")
    parser.add_argument("--model_b_checkpoint", type=str, required=True, help="Чекпоинт модели B")
    parser.add_argument("--test_data", type=str, required=True, help="Путь к тестовым данным")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="comparison")
    args = parser.parse_args()

    compare_models(
        args.model_a_checkpoint,
        args.model_b_checkpoint,
        args.test_data,
        args.num_points,
        args.batch_size,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
