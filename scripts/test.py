"""
Скрипт оценки обученного checkpoint на test-наборе.

Поддерживает обе задачи:
- segmentation,
- classification.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torch import amp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LiDARDataset
from src.models import (
    PointNetSegmentation,
    PointNetClassification,
    PointNetPlusPlusSegmentation,
    PointNetPlusPlusClassification,
    DGCNNSegmentation,
    DGCNNClassification,
    LDGCNNSegmentation,
    LDGCNNClassification,
    build_model,
)
from src.utils.metrics import calculate_metrics


def autocast_context(use_amp):
    if not use_amp:
        return torch.no_grad()

    class _AutocastNoGradContext:
        def __enter__(self):
            self._no_grad = torch.no_grad()
            self._no_grad.__enter__()
            try:
                self._autocast = amp.autocast(device_type="cuda", enabled=True)
            except TypeError:
                self._autocast = amp.autocast(enabled=True)
            return self._autocast.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            autocast_result = self._autocast.__exit__(exc_type, exc_val, exc_tb)
            no_grad_result = self._no_grad.__exit__(exc_type, exc_val, exc_tb)
            return autocast_result or no_grad_result

    return _AutocastNoGradContext()


def resolve_test_path(args):
    if args.test_data:
        return args.test_data
    data_root = args.data_root or os.path.join("Files", args.dataset, "LiDAR")
    stem = f"{args.dataset}_test"
    # GroundTruth-файл содержит метки классов — нужен для evaluation.
    # Обычный test-файл может не иметь Classification → использоваться только для predictions.
    for candidate_stem in [f"{stem}_GroundTruth", stem]:
        for ext in (".laz", ".las", ".txt"):
            candidate = os.path.join(data_root, candidate_stem + ext)
            if os.path.exists(candidate):
                return candidate
    return os.path.join(data_root, stem + ".laz")


def build_test_run_name(args, model_type, task, checkpoint):
    if args.run_name:
        return args.run_name

    parts = [model_type, task, "test", args.dataset.lower()]
    if model_type == "ldgcnn":
        attention_type = checkpoint.get("attention_type", "none")
        if attention_type != "none":
            parts.append(f"attn-{attention_type}")
    loss_type = checkpoint.get("loss_type", "ce")
    if loss_type != "ce":
        parts.append(loss_type)
    class_balance = checkpoint.get("class_balance", "none")
    if class_balance != "none":
        parts.append(f"cb-{class_balance}")
    parts.append(Path(args.checkpoint).stem)
    return "__".join(parts)


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
    parser.add_argument("--run_name", type=str, default=None, help="Имя запуска в MLflow")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Устройство для инференса")
    parser.add_argument("--amp", action="store_true", help="Включить mixed precision (AMP) на GPU")
    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("CUDA недоступна, переключаюсь на CPU.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Использование устройства: {device}")
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and device.type != "cuda":
        print("AMP запрошен, но CUDA недоступна. Выполняю инференс без AMP.")

    # Из checkpoint берем не только веса, но и метаданные (task/model/num_classes).
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    num_features = checkpoint["num_features"]
    model_type = checkpoint.get("model_type", "pointnet")
    task = checkpoint.get("task", "segmentation")
    attention_type = checkpoint.get("attention_type", "none")
    k = checkpoint.get("k", 20)
    k_small = checkpoint.get("k_small", 20)
    k_large = checkpoint.get("k_large", 40)
    attention_k = checkpoint.get("attention_k", 16)
    attention_heads = checkpoint.get("attention_heads", 4)
    attention_dropout = checkpoint.get("attention_dropout", 0.1)
    pt_k = checkpoint.get("pt_k", 16)
    class_to_idx = checkpoint.get("class_to_idx")
    normalize_stats = checkpoint.get("normalize_stats")

    model = build_model(
        model_type,
        task,
        num_classes,
        num_features,
        k=k,
        k_small=k_small,
        k_large=k_large,
        attention_type=attention_type,
        attention_k=attention_k,
        attention_heads=attention_heads,
        attention_dropout=attention_dropout,
        pt_k=pt_k,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_path = resolve_test_path(args)
    print(f"Тестовые данные: {test_path}")

    if normalize_stats is None:
        print("Предупреждение: normalize_stats отсутствуют в чекпоинте. "
              "Нормализация Intensity будет вычислена из тестовых данных.")
    test_dataset = LiDARDataset(
        test_path,
        num_points=args.num_points,
        augment=False,
        has_labels=True,
        task=task,
        class_to_idx=class_to_idx,
        normalize_stats=normalize_stats,
    )
    test_dataset.num_classes = num_classes
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Для сегментации метрики считаем на уровне исходной сцены после
    # voting по перекрывающимся окнам (как в predictions.py). Без voting
    # mIoU/Accuracy оценивают модель на случайно нарезанных блоках, что
    # методологически некорректно: на перекрытиях точки классифицируются
    # многократно, и пограничные случаи доминируют в метриках.
    use_voting = (
        task == "segmentation"
        and not getattr(test_dataset, "chunked", False)
        and getattr(test_dataset, "labels", None) is not None
        and getattr(test_dataset, "features", None) is not None
    )
    if task == "segmentation" and not use_voting:
        print(
            "Внимание: voting по перекрытиям недоступен (chunked-датасет или "
            "отсутствуют labels/features). Метрики считаются по блокам."
        )

    vote_counts = None
    if use_voting:
        num_total_points = len(test_dataset.features)
        vote_counts = np.zeros((num_total_points, num_classes), dtype=np.int32)

    all_predictions = []
    all_targets = []
    total_points = 0
    total_batches = 0
    sample_offset = 0
    inference_start = time.perf_counter()
    with autocast_context(use_amp):
        for features, labels in test_loader:
            features = features.float().to(device)
            labels = labels.long().to(device)

            batch_start = time.perf_counter()
            if task == "classification":
                predictions = model(features)
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                if model_type == "pointnet":
                    predictions, _, _ = model(features)
                else:
                    predictions = model(features)
                pred_classes = torch.argmax(predictions, dim=2)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = time.perf_counter() - batch_start

            if use_voting:
                pred_np = pred_classes.cpu().numpy()
                batch_size = pred_np.shape[0]
                for b in range(batch_size):
                    cloud_indices = test_dataset.get_cloud_point_indices(
                        sample_offset + b
                    )
                    np.add.at(vote_counts, (cloud_indices, pred_np[b]), 1)
                sample_offset += batch_size
                total_points += labels.numel()
            else:
                all_predictions.append(pred_classes)
                all_targets.append(labels)
                if task == "classification":
                    total_points += labels.shape[0]
                else:
                    total_points += labels.numel()
            total_batches += 1

    inference_time = time.perf_counter() - inference_start

    if use_voting:
        covered_mask = vote_counts.sum(axis=1) > 0
        coverage_ratio = float(covered_mask.sum()) / max(1, len(covered_mask))
        print(f"Покрытие точек voting-ом: {coverage_ratio * 100:.2f}%")

        # Пропуск непокрытых точек искажает метрики (занижает знаменатель
        # в IoU, прячет пробелы в нарезке). Правильное поведение:
        #  - при полном покрытии считать по всем точкам;
        #  - при неполном — жёстко падать, чтобы бага нарезки не маскировалась.
        # Dataset (`_create_point_clouds`) гарантирует 100% coverage, поэтому
        # штатно этот assert не должен срабатывать.
        if not covered_mask.all():
            num_missing = int((~covered_mask).sum())
            raise RuntimeError(
                f"Voting не покрыл {num_missing} точек из {len(covered_mask)}. "
                "Это указывает на проблему нарезки в LiDARDataset "
                "(_create_point_clouds должен давать 100% покрытие)."
            )

        aggregated_preds = vote_counts.argmax(axis=1)
        true_labels = np.asarray(test_dataset.labels, dtype=np.int64)

        preds_tensor = torch.from_numpy(aggregated_preds.astype(np.int64))
        targets_tensor = torch.from_numpy(true_labels)
        metrics = calculate_metrics(
            preds_tensor, targets_tensor, num_classes, task="segmentation", include_report=True
        )
    else:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets, num_classes, task=task, include_report=True)

    print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    if task == "segmentation":
        print(f"mIoU: {metrics['mean_iou']:.4f}")
    if inference_time > 0:
        print(f"Inference time: {inference_time:.2f} сек")
        print(f"Points per sec: {total_points / inference_time:,.0f}")

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n")
            f.write("=" * 50 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            if task == "segmentation":
                f.write(f"mIoU: {metrics['mean_iou']:.4f}\n")
            if inference_time > 0:
                f.write(f"Inference time: {inference_time:.2f} сек\n")
                f.write(f"Points per sec: {total_points / inference_time:,.0f}\n")

    mlflow.set_experiment(args.experiment_name)
    run_name = build_test_run_name(args, model_type, task, checkpoint)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": model_type,
                "task": task,
                "dataset": args.dataset,
                "attention_type": attention_type,
                "attention_k": attention_k,
                "attention_heads": attention_heads,
                "attention_dropout": attention_dropout,
                "amp": use_amp,
                "run_name": run_name,
            }
        )
        mlflow.log_param("voting", bool(use_voting))
        mlflow.log_metric("test_accuracy", metrics["accuracy"])
        if task == "segmentation":
            mlflow.log_metric("test_miou", metrics["mean_iou"])
        if use_voting:
            mlflow.log_metric("voting_coverage", coverage_ratio)
        if inference_time > 0:
            mlflow.log_metric("inference_time_sec", inference_time)
            mlflow.log_metric("points_per_sec", total_points / inference_time)


if __name__ == "__main__":
    main()
