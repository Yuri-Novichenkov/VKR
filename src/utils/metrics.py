"""
Функции вычисления метрик для задач сегментации и классификации.

Единый модуль используется во всех скриптах (train.py, test.py),
чтобы исключить расхождение формул при копировании.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def calculate_metrics(
    predictions,
    targets,
    num_classes: int,
    task: str = "segmentation",
    include_report: bool = False,
) -> dict:
    """Вычисляет стандартный набор метрик для задачи сегментации или классификации.

    Args:
        predictions: тензор или массив предсказанных меток.
        targets:     тензор или массив истинных меток.
        num_classes: общее число классов (для построения confusion matrix).
        task:        "segmentation" | "classification".
        include_report: если True — добавляет classification_report в словарь.
                        По умолчанию False (для обучения не нужен).

    Returns:
        Словарь с ключами: accuracy, confusion_matrix, и для segmentation —
        mean_iou, per_class_iou; опционально — classification_report.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    labels_range = list(range(num_classes))

    if task == "classification":
        accuracy = accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions, labels=labels_range)
        result = {
            "accuracy": float(accuracy),
            "confusion_matrix": cm,
        }
        if include_report:
            result["classification_report"] = classification_report(
                targets,
                predictions,
                labels=labels_range,
                output_dict=True,
                zero_division=0,
            )
        return result

    # Segmentation: метрики считаются на уровне отдельных точек.
    flat_pred = predictions.flatten()
    flat_tgt = targets.flatten()

    accuracy = accuracy_score(flat_tgt, flat_pred)
    cm = confusion_matrix(flat_tgt, flat_pred, labels=labels_range)

    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        ious.append(float(intersection / union) if union > 0 else 0.0)

    mean_iou = float(np.mean(ious))

    result = {
        "accuracy": float(accuracy),
        "mean_iou": mean_iou,
        "per_class_iou": ious,
        "confusion_matrix": cm,
    }
    if include_report:
        result["classification_report"] = classification_report(
            flat_tgt,
            flat_pred,
            labels=labels_range,
            output_dict=True,
            zero_division=0,
        )
    return result
