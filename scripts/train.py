"""
Скрипт обучения моделей point-cloud задач:
- segmentation (поточечная классификация),
- classification (класс облака-окна).

Скрипт отвечает за:
- подготовку датасета/loader'ов,
- выбор архитектуры,
- цикл train/val,
- сохранение checkpoint'ов,
- логирование в MLflow.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from functools import partial
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from contextlib import contextmanager
from torch.utils.data import DataLoader
from tqdm import tqdm

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@contextmanager
def autocast_context(use_amp):
    if not use_amp:
        yield
        return
    try:
        with amp.autocast(device_type="cuda", enabled=True):
            yield
    except TypeError:
        with amp.autocast(enabled=True):
            yield


def make_scaler(use_amp):
    if not use_amp:
        return None
    try:
        return amp.GradScaler(device_type="cuda", enabled=True)
    except TypeError:
        return amp.GradScaler(enabled=True)


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    num_classes,
    model_type,
    task="segmentation",
    lambda_reg=0.001,
    use_amp=False,
    scaler=None,
    class_weights=None,
    loss_type="ce",
    focal_gamma=2.0,
):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    pbar = tqdm(train_loader, desc="Training")
    for features, labels in pbar:
        features = features.float().to(device)
        labels = labels.long().to(device)

        with autocast_context(use_amp):
            if task == "classification":
                predictions = model(features)
                if loss_type == "ce":
                    loss, ce_loss, reg_loss = model.get_loss(
                        predictions,
                        labels,
                        class_weights=class_weights,
                    )
                elif loss_type in ("focal", "cb_focal"):
                    ce_per_sample = F.cross_entropy(
                        predictions,
                        labels,
                        weight=class_weights,
                        reduction="none",
                    )
                    pt = torch.exp(-ce_per_sample)
                    ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                    loss = ce_loss
                    reg_loss = torch.tensor(0.0, device=predictions.device)
                else:
                    raise ValueError(f"Неизвестный loss_type: {loss_type} (lovasz не поддерживается для classification)")
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                if model_type == "pointnet":
                    predictions, transform_coords, transform_features = model(features)
                    if loss_type == "ce":
                        loss, ce_loss, reg_loss = model.get_loss(
                            predictions,
                            labels,
                            transform_coords,
                            transform_features,
                            lambda_reg=lambda_reg,
                            class_weights=class_weights,
                        )
                    elif loss_type in ("focal", "cb_focal"):
                        bsz, n_points, n_classes = predictions.shape
                        predictions_flat = predictions.reshape(-1, n_classes)
                        labels_flat = labels.reshape(-1)
                        ce_per_sample = F.cross_entropy(
                            predictions_flat,
                            labels_flat,
                            weight=class_weights,
                            reduction="none",
                        )
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                        identity_3 = torch.eye(3, device=transform_coords.device).unsqueeze(0)
                        reg_coords = torch.mean(
                            torch.norm(
                                torch.bmm(transform_coords, transform_coords.transpose(2, 1)) - identity_3,
                                dim=(1, 2),
                            )
                        )
                        identity_64 = torch.eye(64, device=transform_features.device).unsqueeze(0)
                        reg_features = torch.mean(
                            torch.norm(
                                torch.bmm(transform_features, transform_features.transpose(2, 1)) - identity_64,
                                dim=(1, 2),
                            )
                        )
                        reg_loss = reg_coords + reg_features
                        loss = ce_loss + lambda_reg * reg_loss
                    elif loss_type == "lovasz":
                        try:
                            from lovász_losses import lovász_softmax
                        except (ImportError, ModuleNotFoundError):
                            from lovasz_losses import lovasz_softmax as lovász_softmax
                        bsz, n_points, n_classes = predictions.shape
                        probs = torch.softmax(predictions, dim=2).permute(0, 2, 1).contiguous()  # (B, C, N)
                        ce_loss = lovász_softmax(probs, labels, classes="present")
                        if class_weights is not None:
                            # Вспомогательный взвешенный CE для усиления балансировки
                            flat_p = predictions.reshape(-1, n_classes)
                            flat_l = labels.reshape(-1)
                            aux = F.cross_entropy(flat_p, flat_l, weight=class_weights)
                            ce_loss = ce_loss + 0.5 * aux
                        identity_3 = torch.eye(3, device=transform_coords.device).unsqueeze(0)
                        reg_coords = torch.mean(
                            torch.norm(
                                torch.bmm(transform_coords, transform_coords.transpose(2, 1)) - identity_3,
                                dim=(1, 2),
                            )
                        )
                        identity_64 = torch.eye(64, device=transform_features.device).unsqueeze(0)
                        reg_features = torch.mean(
                            torch.norm(
                                torch.bmm(transform_features, transform_features.transpose(2, 1)) - identity_64,
                                dim=(1, 2),
                            )
                        )
                        reg_loss = reg_coords + reg_features
                        loss = ce_loss + lambda_reg * reg_loss
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type}")
                else:
                    predictions = model(features)
                    if loss_type == "ce":
                        loss, ce_loss, reg_loss = model.get_loss(
                            predictions,
                            labels,
                            class_weights=class_weights,
                        )
                    elif loss_type in ("focal", "cb_focal"):
                        bsz, n_points, n_classes = predictions.shape
                        predictions_flat = predictions.reshape(-1, n_classes)
                        labels_flat = labels.reshape(-1)
                        ce_per_sample = F.cross_entropy(
                            predictions_flat,
                            labels_flat,
                            weight=class_weights,
                            reduction="none",
                        )
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                        loss = ce_loss
                        reg_loss = torch.tensor(0.0, device=predictions.device)
                    elif loss_type == "lovasz":
                        try:
                            from lovász_losses import lovász_softmax
                        except (ImportError, ModuleNotFoundError):
                            from lovasz_losses import lovasz_softmax as lovász_softmax
                        bsz, n_points, n_classes = predictions.shape
                        probs = torch.softmax(predictions, dim=2).permute(0, 2, 1).contiguous()  # (B, C, N)
                        ce_loss = lovász_softmax(probs, labels, classes="present")
                        if class_weights is not None:
                            flat_p = predictions.reshape(-1, n_classes)
                            flat_l = labels.reshape(-1)
                            aux = F.cross_entropy(flat_p, flat_l, weight=class_weights)
                            ce_loss = ce_loss + 0.5 * aux
                        loss = ce_loss
                        reg_loss = torch.tensor(0.0, device=predictions.device)
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type}")
                pred_classes = torch.argmax(predictions, dim=2)

        # NaN-guard: ловим расхождение как можно раньше. Без этого молчаливый
        # NaN (например, при AMP + ненормализованных координатах PointNet++)
        # проходит всю тренировку с return_code=0 и даёт бесполезный чекпоинт.
        if not torch.isfinite(loss):
            ce_val = float(ce_loss.item()) if ce_loss is not None else float("nan")
            reg_val = float(reg_loss.item()) if (reg_loss is not None and reg_loss.numel() > 0) else 0.0
            raise RuntimeError(
                "Train loss is not finite "
                f"(loss={loss.item()}, ce_loss={ce_val}, reg_loss={reg_val}). "
                "Тренировка расходится — вероятные причины: AMP + ненормализованные координаты, "
                "слишком большой lr, неправильные радиусы PointNet++, взорвавшиеся class_weights."
            )

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # unscale перед clip, чтобы норма считалась в исходных единицах.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        all_predictions.append(pred_classes)
        all_targets.append(labels)

        postfix = {"loss": f"{loss.item():.4f}"}
        if ce_loss is not None:
            postfix["ce_loss"] = f"{ce_loss.item():.4f}"
        if reg_loss is not None and reg_loss.numel() > 0:
            postfix["reg_loss"] = f"{reg_loss.item():.4f}"
        pbar.set_postfix(postfix)

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_classes, task=task)
    metrics["loss"] = total_loss / len(train_loader)
    return metrics


def validate(
    model,
    val_loader,
    device,
    num_classes,
    model_type,
    task="segmentation",
    use_amp=False,
    class_weights=None,
    loss_type="ce",
    focal_gamma=2.0,
    lambda_reg=0.001,
    val_dataset=None,
):
    """
    Выбор метрик:
    - segmentation: если передан val_dataset и он не chunked, то для
      оценки используется scene-level voting по перекрывающимся окнам
      (как в test.py / predictions.py). Иначе — block-level.
    - classification: всегда block-level (один label на окно).
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    # Включаем voting только если: задача — сегментация, датасет передан,
    # не chunked и предоставляет labels/features в памяти. shuffle=False
    # для val_loader уже гарантируется в main().
    use_voting = (
        task == "segmentation"
        and val_dataset is not None
        and not getattr(val_dataset, "chunked", False)
        and getattr(val_dataset, "labels", None) is not None
        and getattr(val_dataset, "features", None) is not None
    )
    vote_counts = None
    sample_offset = 0
    if use_voting:
        num_total_points = len(val_dataset.features)
        vote_counts = np.zeros((num_total_points, num_classes), dtype=np.int32)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for features, labels in pbar:
            features = features.float().to(device)
            labels = labels.long().to(device)

            with autocast_context(use_amp):
                if task == "classification":
                    predictions = model(features)
                    if loss_type == "ce":
                        loss, ce_loss, reg_loss = model.get_loss(
                            predictions,
                            labels,
                            class_weights=class_weights,
                        )
                    elif loss_type in ("focal", "cb_focal"):
                        ce_per_sample = F.cross_entropy(
                            predictions,
                            labels,
                            weight=class_weights,
                            reduction="none",
                        )
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                        loss = ce_loss
                        reg_loss = torch.tensor(0.0, device=predictions.device)
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type} (lovasz не поддерживается для classification)")
                    pred_classes = torch.argmax(predictions, dim=1)
                else:
                    if model_type == "pointnet":
                        predictions, transform_coords, transform_features = model(features)
                        if loss_type == "ce":
                            loss, ce_loss, reg_loss = model.get_loss(
                                predictions,
                                labels,
                                transform_coords,
                                transform_features,
                                lambda_reg=lambda_reg,
                                class_weights=class_weights,
                            )
                        elif loss_type in ("focal", "cb_focal"):
                            bsz, n_points, n_classes = predictions.shape
                            predictions_flat = predictions.reshape(-1, n_classes)
                            labels_flat = labels.reshape(-1)
                            ce_per_sample = F.cross_entropy(
                                predictions_flat,
                                labels_flat,
                                weight=class_weights,
                                reduction="none",
                            )
                            pt = torch.exp(-ce_per_sample)
                            ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                            identity_3 = torch.eye(3, device=transform_coords.device).unsqueeze(0)
                            reg_coords = torch.mean(
                                torch.norm(
                                    torch.bmm(transform_coords, transform_coords.transpose(2, 1)) - identity_3,
                                    dim=(1, 2),
                                )
                            )
                            identity_64 = torch.eye(64, device=transform_features.device).unsqueeze(0)
                            reg_features = torch.mean(
                                torch.norm(
                                    torch.bmm(transform_features, transform_features.transpose(2, 1)) - identity_64,
                                    dim=(1, 2),
                                )
                            )
                            reg_loss = reg_coords + reg_features
                            loss = ce_loss + lambda_reg * reg_loss
                        elif loss_type == "lovasz":
                            try:
                                from lovász_losses import lovász_softmax
                            except (ImportError, ModuleNotFoundError):
                                from lovasz_losses import lovasz_softmax as lovász_softmax
                            bsz, n_points, n_classes = predictions.shape
                            probs = torch.softmax(predictions, dim=2).permute(0, 2, 1).contiguous()  # (B, C, N)
                            ce_loss = lovász_softmax(probs, labels, classes="present")
                            if class_weights is not None:
                                flat_p = predictions.reshape(-1, n_classes)
                                flat_l = labels.reshape(-1)
                                aux = F.cross_entropy(flat_p, flat_l, weight=class_weights)
                                ce_loss = ce_loss + 0.5 * aux
                            identity_3 = torch.eye(3, device=transform_coords.device).unsqueeze(0)
                            reg_coords = torch.mean(
                                torch.norm(
                                    torch.bmm(transform_coords, transform_coords.transpose(2, 1)) - identity_3,
                                    dim=(1, 2),
                                )
                            )
                            identity_64 = torch.eye(64, device=transform_features.device).unsqueeze(0)
                            reg_features = torch.mean(
                                torch.norm(
                                    torch.bmm(transform_features, transform_features.transpose(2, 1)) - identity_64,
                                    dim=(1, 2),
                                )
                            )
                            reg_loss = reg_coords + reg_features
                            loss = ce_loss + lambda_reg * reg_loss
                        else:
                            raise ValueError(f"Неизвестный loss_type: {loss_type}")
                    else:
                        predictions = model(features)
                        if loss_type == "ce":
                            loss, ce_loss, reg_loss = model.get_loss(
                                predictions,
                                labels,
                                class_weights=class_weights,
                            )
                        elif loss_type in ("focal", "cb_focal"):
                            bsz, n_points, n_classes = predictions.shape
                            predictions_flat = predictions.reshape(-1, n_classes)
                            labels_flat = labels.reshape(-1)
                            ce_per_sample = F.cross_entropy(
                                predictions_flat,
                                labels_flat,
                                weight=class_weights,
                                reduction="none",
                            )
                            pt = torch.exp(-ce_per_sample)
                            ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                            loss = ce_loss
                            reg_loss = torch.tensor(0.0, device=predictions.device)
                        elif loss_type == "lovasz":
                            try:
                                from lovász_losses import lovász_softmax
                            except (ImportError, ModuleNotFoundError):
                                from lovasz_losses import lovasz_softmax as lovász_softmax
                            bsz, n_points, n_classes = predictions.shape
                            probs = torch.softmax(predictions, dim=2).permute(0, 2, 1).contiguous()  # (B, C, N)
                            ce_loss = lovász_softmax(probs, labels, classes="present")
                            if class_weights is not None:
                                flat_p = predictions.reshape(-1, n_classes)
                                flat_l = labels.reshape(-1)
                                aux = F.cross_entropy(flat_p, flat_l, weight=class_weights)
                                ce_loss = ce_loss + 0.5 * aux
                            loss = ce_loss
                            reg_loss = torch.tensor(0.0, device=predictions.device)
                        else:
                            raise ValueError(f"Неизвестный loss_type: {loss_type}")
                    pred_classes = torch.argmax(predictions, dim=2)

            total_loss += loss.item()

            if use_voting:
                pred_np = pred_classes.cpu().numpy()
                bsz = pred_np.shape[0]
                for b in range(bsz):
                    cloud_indices = val_dataset.get_cloud_point_indices(
                        sample_offset + b
                    )
                    np.add.at(vote_counts, (cloud_indices, pred_np[b]), 1)
                sample_offset += bsz
            else:
                all_predictions.append(pred_classes)
                all_targets.append(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    if use_voting:
        covered_mask = vote_counts.sum(axis=1) > 0
        if not covered_mask.all():
            num_missing = int((~covered_mask).sum())
            raise RuntimeError(
                f"Voting не покрыл {num_missing} точек из {len(covered_mask)} "
                "на валидации. Dataset._create_point_clouds должен давать "
                "100% покрытие (проверить spatial slicing)."
            )
        aggregated_preds = vote_counts.argmax(axis=1)
        true_labels = np.asarray(val_dataset.labels, dtype=np.int64)
        preds_tensor = torch.from_numpy(aggregated_preds.astype(np.int64))
        targets_tensor = torch.from_numpy(true_labels)
        metrics = calculate_metrics(
            preds_tensor, targets_tensor, num_classes, task="segmentation"
        )
    else:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_predictions, all_targets, num_classes, task=task)
    metrics["loss"] = total_loss / len(val_loader)
    return metrics


def resolve_data_paths(args):
    if args.train_data and args.val_data:
        return args.train_data, args.val_data

    # Если пути не переданы явно, используем соглашение проекта.
    data_root = args.data_root
    if data_root is None:
        data_root = os.path.join("Files", args.dataset, "LiDAR")

    train_data = os.path.join(data_root, f"{args.dataset}_train.txt")
    val_data = os.path.join(data_root, f"{args.dataset}_val.txt")
    return train_data, val_data


def set_seed(seed, deterministic=False):
    """
    Фиксирует источники случайности.
    deterministic=True включает более строгую воспроизводимость ценой скорости.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Для воспроизводимости отключаем недетерминированные fast-path режимы.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Для совместимости со старыми версиями PyTorch без warn_only.
            torch.use_deterministic_algorithms(True)
    else:
        # Возвращаем быстрые backend-оптимизации. На GPU это обычно заметно
        # ускоряет обучение по сравнению со строгим детерминизмом.
        torch.backends.cudnn.deterministic = False
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except TypeError:
            pass


def seed_worker_with_base(worker_id: int, base_seed: int):
    """
    Инициализирует RNG DataLoader worker детерминированным seed.
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def compute_class_weights(train_dataset, num_classes, task="segmentation", mode="none", beta=0.999):
    """
    Считает веса классов по train-набору.
    mode:
      - none: без балансировки
      - inverse: w_c = 1 / count_c
      - effective: Class-Balanced Loss (Cui et al.), w_c = (1-beta)/(1-beta^n_c)
    """
    if mode == "none":
        return None

    counts = np.zeros(num_classes, dtype=np.int64)
    if task == "segmentation":
        # Проходим по выборке через __getitem__, чтобы работало и для chunked-кэша.
        for i in range(len(train_dataset)):
            _features, labels = train_dataset[i]
            counts += np.bincount(labels.numpy(), minlength=num_classes)
    else:
        for i in range(len(train_dataset)):
            _features, label = train_dataset[i]
            counts[int(label.item())] += 1

    counts = np.maximum(counts, 1)
    if mode == "inverse":
        weights = 1.0 / counts.astype(np.float64)
    elif mode == "effective":
        effective_num = 1.0 - np.power(beta, counts.astype(np.float64))
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    else:
        raise ValueError(f"Неизвестный class_balance режим: {mode}")

    # Нормируем к среднему весу 1.0 для стабильной оптимизации.
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def build_train_run_name(args):
    if args.run_name:
        return args.run_name

    parts = [args.model, args.task, args.dataset.lower()]
    if args.model == "ldgcnn" and args.attention_type != "none":
        parts.append(f"attn-{args.attention_type}")
    if args.loss_type != "ce":
        parts.append(args.loss_type)
    if args.class_balance != "none":
        parts.append(f"cb-{args.class_balance}")
    parts.append(f"pts{args.num_points}")
    parts.append(f"bs{args.batch_size}")
    parts.append(f"seed{args.seed}")
    return "__".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Обучение моделей для сегментации/классификации")
    parser.add_argument("--train_data", type=str, default=None, help="Путь к обучающему набору")
    parser.add_argument("--val_data", type=str, default=None, help="Путь к валидационному набору")
    parser.add_argument("--data_root", type=str, default=None, help="Корневая папка с данными (например Files/Mar18/LiDAR)")
    parser.add_argument("--dataset", type=str, default="Mar16", help="Префикс датасета (Mar16 или Mar18)")
    parser.add_argument("--num_points", type=int, default=4096, help="Количество точек в облаке")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--epochs", type=int, default=100, help="Количество эпох")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="step",
        choices=["step", "cosine"],
        help="Планировщик lr: step (StepLR, шаг 20 эпох, gamma 0.7) или cosine (CosineAnnealingLR до lr*0.01)",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="Количество классов")
    parser.add_argument("--lambda_reg", type=float, default=0.001, help="Коэффициент регуляризации трансформаций")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Директория для сохранения моделей")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для возобновления обучения")
    parser.add_argument("--model", type=str, default="pointnet", choices=["pointnet", "pointnet++", "dgcnn", "ldgcnn"], help="Модель")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "classification"], help="Задача")
    parser.add_argument("--experiment_name", type=str, default="PointCloudExperiments", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="Имя запуска в MLflow")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (для Windows лучше 0)")
    parser.add_argument("--amp", action="store_true", help="Включить mixed precision (AMP) на GPU")
    parser.add_argument("--k", type=int, default=20, help="k для DGCNN")
    parser.add_argument("--k_small", type=int, default=20, help="k_small для LDGCNN")
    parser.add_argument("--k_large", type=int, default=40, help="k_large для LDGCNN")
    parser.add_argument(
        "--attention_type",
        type=str,
        default="none",
        choices=["none", "gatv2", "local_window"],
        help="Тип attention для LDGCNN",
    )
    parser.add_argument("--attention_k", type=int, default=16, help="Размер локального окна attention")
    parser.add_argument("--attention_heads", type=int, default=4, help="Количество attention heads")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Dropout в attention")
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal", "cb_focal", "lovasz"], help="Тип функции потерь")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma для focal loss")
    parser.add_argument(
        "--class_balance",
        type=str,
        default="none",
        choices=["none", "inverse", "effective"],
        help="Режим балансировки классов через веса в CrossEntropy",
    )
    parser.add_argument("--class_balance_beta", type=float, default=0.999, help="Параметр beta для режима effective")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Директория для кэша npz")
    parser.add_argument("--cache_mode", type=str, default="write", choices=["off", "read", "write"], help="Режим кэша")
    parser.add_argument("--cache_chunked", action="store_true", help="Сохранять нарезанные облака чанками")
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка по облакам")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="prefetch_factor для DataLoader")
    parser.add_argument("--persistent_workers", action="store_true", help="persistent_workers для DataLoader")
    parser.add_argument("--cache_only", action="store_true", help="Только подготовить кэш и выйти")
    parser.add_argument("--allow_windows_workers", action="store_true", help="Разрешить num_workers>0 на Windows")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимости")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Ранний останов: число эпох без улучшения (0 отключает early stopping).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Минимальное улучшение val-метрики, чтобы считать эпоху 'лучше'.",
    )
    parser.add_argument(
        "--metrics_json_path",
        type=str,
        default=None,
        help="Опциональный путь для сохранения подробных метрик запуска в JSON",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Включить строгую воспроизводимость (обычно медленнее на GPU)",
    )

    args = parser.parse_args()
    if args.loss_type == "cb_focal" and args.class_balance == "none":
        raise ValueError("Для --loss_type cb_focal необходимо включить --class_balance (inverse/effective).")
    if args.loss_type == "lovasz" and args.task == "classification":
        raise ValueError("--loss_type lovasz поддерживается только для задачи segmentation.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)
    set_seed(args.seed, deterministic=args.deterministic)
    logger.info("Seed: %s", args.seed)
    logger.info("Deterministic mode: %s", args.deterministic)

    if os.name == "nt" and args.num_workers > 0 and not args.allow_windows_workers:
        logger.warning("Windows: num_workers>0 может приводить к ошибкам. Устанавливаю num_workers=0.")
        args.num_workers = 0

    # Единый формат хранения checkpoint'ов:
    # checkpoints/<model>/<task>/<variant>/<dataset>/...
    # variant формируется из активных режимов (attention/loss/class_balance),
    # чтобы эксперименты не перезаписывали друг друга.
    if args.save_dir == "checkpoints":
        variant_parts = []
        if args.model == "ldgcnn" and args.attention_type != "none":
            attn_dropout_str = str(args.attention_dropout).replace(".", "p")
            variant_parts.append(
                f"attn_{args.attention_type}_k{args.attention_k}_h{args.attention_heads}_d{attn_dropout_str}"
            )
        if args.loss_type != "ce":
            gamma_str = str(args.focal_gamma).replace(".", "p")
            variant_parts.append(f"loss_{args.loss_type}_g{gamma_str}")
        if args.class_balance != "none":
            beta_str = str(args.class_balance_beta).replace(".", "p")
            variant_parts.append(f"cb_{args.class_balance}_b{beta_str}")

        if variant_parts:
            variant = "__".join(variant_parts)
            args.save_dir = os.path.join(args.save_dir, args.model, args.task, variant, args.dataset.lower())
        else:
            args.save_dir = os.path.join(args.save_dir, args.model, args.task, args.dataset.lower())
    os.makedirs(args.save_dir, exist_ok=True)

    train_data, val_data = resolve_data_paths(args)
    logger.info("Train data: %s", train_data)
    logger.info("Val data: %s", val_data)

    train_dataset = LiDARDataset(
        train_data,
        num_points=args.num_points,
        augment=True,
        task=args.task,
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
        cache_chunked=args.cache_chunked,
        chunk_size=args.chunk_size,
    )
    val_dataset = LiDARDataset(
        val_data,
        num_points=args.num_points,
        augment=False,
        task=args.task,
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
        cache_chunked=args.cache_chunked,
        chunk_size=args.chunk_size,
        class_to_idx=train_dataset.class_to_idx,
        normalize_stats=train_dataset.normalize_stats,
    )

    if args.cache_only:
        logger.info("Кэш подготовлен. Завершение без обучения (--cache_only).")
        return

    num_classes = train_dataset.num_classes if args.num_classes is None else args.num_classes
    logger.info("Количество классов: %s", num_classes)
    class_weights = compute_class_weights(
        train_dataset,
        num_classes=num_classes,
        task=args.task,
        mode=args.class_balance,
        beta=args.class_balance_beta,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info("Включена балансировка классов: %s", args.class_balance)
        logger.info("Class weights: %s", class_weights.detach().cpu().numpy())

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True if torch.cuda.is_available() else False,
        "drop_last": True,
    }
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader_kwargs["worker_init_fn"] = partial(seed_worker_with_base, base_seed=args.seed)
    train_loader_kwargs["generator"] = loader_generator
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader_kwargs["persistent_workers"] = args.persistent_workers

    # Для Windows по умолчанию num_workers=0 (см. аргументы/guard выше).
    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    val_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True if torch.cuda.is_available() else False,
        "drop_last": False,
    }
    val_loader_kwargs["worker_init_fn"] = partial(seed_worker_with_base, base_seed=args.seed + 10_000)
    val_loader_kwargs["generator"] = loader_generator
    if args.num_workers > 0:
        val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader_kwargs["persistent_workers"] = args.persistent_workers

    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    num_features = len(train_dataset.use_features)
    model = build_model(
        args.model,
        args.task,
        num_classes,
        num_features,
        k=args.k,
        k_small=args.k_small,
        k_large=args.k_large,
        attention_type=args.attention_type,
        attention_k=args.attention_k,
        attention_heads=args.attention_heads,
        attention_dropout=args.attention_dropout,
    ).to(device)
    logger.info("Используется модель: %s (%s)", args.model, args.task)
    logger.info("Параметров: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
        logger.info("LR scheduler: CosineAnnealingLR (T_max=%d, eta_min=%.2e)", args.epochs, args.lr * 0.01)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        logger.info("LR scheduler: StepLR (step_size=20, gamma=0.7)")

    start_epoch = 0
    best_val_metric = 0
    best_val_per_class_iou = None

    if args.resume:
        logger.info("Загрузка чекпоинта из %s", args.resume)
        checkpoint = torch.load(args.resume, weights_only=False)
        checkpoint_model_type = checkpoint.get("model_type", args.model)
        checkpoint_task = checkpoint.get("task", args.task)
        checkpoint_attention_type = checkpoint.get("attention_type", "none")
        checkpoint_k = checkpoint.get("k", args.k)
        checkpoint_k_small = checkpoint.get("k_small", args.k_small)
        checkpoint_k_large = checkpoint.get("k_large", args.k_large)
        if checkpoint_model_type != args.model:
            logger.warning(
                "Модель в чекпоинте (%s) не совпадает с аргументом (%s)",
                checkpoint_model_type,
                args.model,
            )
        if checkpoint_task != args.task:
            logger.warning(
                "Task в чекпоинте (%s) не совпадает с аргументом (%s)",
                checkpoint_task,
                args.task,
            )
        if checkpoint_attention_type != args.attention_type:
            logger.warning(
                "attention_type в чекпоинте (%s) не совпадает с аргументом (%s)",
                checkpoint_attention_type,
                args.attention_type,
            )
        if checkpoint_k != args.k or checkpoint_k_small != args.k_small or checkpoint_k_large != args.k_large:
            logger.warning(
                "Параметры графа в чекпоинте не совпадают с аргументами "
                "(checkpoint: k=%s, k_small=%s, k_large=%s; args: k=%s, k_small=%s, k_large=%s)",
                checkpoint_k,
                checkpoint_k_small,
                checkpoint_k_large,
                args.k,
                args.k_small,
                args.k_large,
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.info("Scheduler восстановлен из чекпоинта (last_epoch=%s)", scheduler.last_epoch)
        else:
            # Чекпоинт старого формата: вручную прокручиваем scheduler до нужной эпохи.
            resumed_epoch = checkpoint.get("epoch", 0)
            for _ in range(resumed_epoch):
                scheduler.step()
            logger.info("Scheduler восстановлен вручную (эпоха %s)", resumed_epoch)
        start_epoch = checkpoint["epoch"]
        # Поддержка старых чекпоинтов, где ключ назывался best_val_iou.
        best_val_metric = checkpoint.get("best_val_metric", checkpoint.get("best_val_iou", 0))

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_scaler(use_amp)

    mlflow.set_experiment(args.experiment_name)
    run_name = build_train_run_name(args)
    with mlflow.start_run(run_name=run_name):
        class_weights_list = (
            [float(x) for x in class_weights.detach().cpu().numpy().tolist()]
            if class_weights is not None
            else None
        )
        mlflow.log_params(
            {
                "model": args.model,
                "task": args.task,
                "dataset": args.dataset,
                "data_root": args.data_root or "auto",
                "num_points": args.num_points,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "lr_scheduler": args.lr_scheduler,
                "num_classes": num_classes,
                "num_workers": args.num_workers,
                "seed": args.seed,
                "deterministic": args.deterministic,
                "amp": use_amp,
                "k": args.k,
                "k_small": args.k_small,
                "k_large": args.k_large,
                "attention_type": args.attention_type,
                "attention_k": args.attention_k,
                "attention_heads": args.attention_heads,
                "attention_dropout": args.attention_dropout,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
                "class_balance": args.class_balance,
                "class_balance_beta": args.class_balance_beta,
                "cache_dir": args.cache_dir,
                "cache_mode": args.cache_mode,
                "cache_chunked": args.cache_chunked,
                "chunk_size": args.chunk_size,
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": args.persistent_workers,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "run_name": run_name,
            }
        )
        if class_weights_list is not None:
            mlflow.log_dict({"class_weights": class_weights_list}, "class_weights.json")

        logger.info("Начало обучения")
        epoch_stats = []
        train_start_ts = time.perf_counter()
        epochs_without_improvement = 0
        for epoch in range(start_epoch, args.epochs):
            logger.info("Эпоха %s/%s", epoch + 1, args.epochs)
            epoch_start_ts = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            train_metrics = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                num_classes,
                args.model,
                task=args.task,
                lambda_reg=args.lambda_reg,
                use_amp=use_amp,
                scaler=scaler,
                class_weights=class_weights,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
            )
            val_metrics = validate(
                model,
                val_loader,
                device,
                num_classes,
                args.model,
                task=args.task,
                use_amp=use_amp,
                class_weights=class_weights,
                loss_type=args.loss_type,
                focal_gamma=args.focal_gamma,
                lambda_reg=args.lambda_reg,
                val_dataset=val_dataset,
            )
            scheduler.step()
            epoch_time_sec = float(time.perf_counter() - epoch_start_ts)
            peak_vram_mb = (
                float(torch.cuda.max_memory_allocated() / (1024 ** 2))
                if torch.cuda.is_available()
                else 0.0
            )

            train_msg = (
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
                + (f", mIoU: {train_metrics['mean_iou']:.4f}" if args.task == "segmentation" else "")
            )
            val_msg = (
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}"
                + (f", mIoU: {val_metrics['mean_iou']:.4f}" if args.task == "segmentation" else "")
            )
            logger.info(train_msg)
            logger.info(val_msg)
            logger.info("Epoch time: %.2fs | Peak VRAM: %.1f MB", epoch_time_sec, peak_vram_mb)

            # MLflow metrics
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            if args.task == "segmentation":
                mlflow.log_metric("train_miou", train_metrics["mean_iou"], step=epoch)
                mlflow.log_metric("val_miou", val_metrics["mean_iou"], step=epoch)
                # Per-class IoU — ключевой сигнал для свипа class_balance_beta:
                # общий mIoU может меняться слабо, а IoU редких классов — сильно.
                for ci, iou_c in enumerate(val_metrics.get("per_class_iou", []) or []):
                    mlflow.log_metric(f"val_iou_c{ci}", float(iou_c), step=epoch)
            mlflow.log_metric("epoch_time_sec", epoch_time_sec, step=epoch)
            mlflow.log_metric("peak_vram_mb", peak_vram_mb, step=epoch)

            epoch_row = {
                "epoch": epoch + 1,
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "epoch_time_sec": epoch_time_sec,
                "peak_vram_mb": peak_vram_mb,
            }
            if args.task == "segmentation":
                epoch_row["train_miou"] = float(train_metrics["mean_iou"])
                epoch_row["val_miou"] = float(val_metrics["mean_iou"])
                epoch_row["train_per_class_iou"] = [float(x) for x in train_metrics.get("per_class_iou", []) or []]
                epoch_row["val_per_class_iou"] = [float(x) for x in val_metrics.get("per_class_iou", []) or []]
            epoch_stats.append(epoch_row)

            # Сохраняем лучшую модель по целевой метрике:
            # mIoU для segmentation, accuracy для classification.
            if args.task == "segmentation":
                metric_for_best = val_metrics["mean_iou"]
            else:
                metric_for_best = val_metrics["accuracy"]

            improved = metric_for_best > (best_val_metric + args.early_stopping_min_delta)
            if improved:
                best_val_metric = metric_for_best
                epochs_without_improvement = 0
                best_val_per_class_iou = (
                    [float(x) for x in val_metrics.get("per_class_iou", []) or []]
                    if args.task == "segmentation"
                    else None
                )
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "num_classes": num_classes,
                    "num_features": num_features,
                    "model_type": args.model,
                    "task": args.task,
                    "attention_type": args.attention_type,
                    "k": args.k,
                    "k_small": args.k_small,
                    "k_large": args.k_large,
                    "attention_k": args.attention_k,
                    "attention_heads": args.attention_heads,
                    "attention_dropout": args.attention_dropout,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                    "class_balance": args.class_balance,
                    "class_balance_beta": args.class_balance_beta,
                    "lr_scheduler": args.lr_scheduler,
                    "class_to_idx": train_dataset.class_to_idx,
                    "idx_to_class": train_dataset.idx_to_class,
                    "normalize_stats": train_dataset.normalize_stats,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                best_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                mlflow.log_artifact(best_path)
                logger.info("Сохранена лучшая модель: %s", best_path)
            else:
                epochs_without_improvement += 1

            # Сохранение последнего чекпоинта
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_metric": best_val_metric,
                "num_classes": num_classes,
                "num_features": num_features,
                "model_type": args.model,
                "task": args.task,
                "attention_type": args.attention_type,
                "k": args.k,
                "k_small": args.k_small,
                "k_large": args.k_large,
                "attention_k": args.attention_k,
                "attention_heads": args.attention_heads,
                "attention_dropout": args.attention_dropout,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
                "class_balance": args.class_balance,
                "class_balance_beta": args.class_balance_beta,
                "class_to_idx": train_dataset.class_to_idx,
                "idx_to_class": train_dataset.idx_to_class,
                "normalize_stats": train_dataset.normalize_stats,
            }
            last_path = os.path.join(args.save_dir, "last_checkpoint.pth")
            torch.save(checkpoint, last_path)

            if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                logger.info(
                    "Early stopping: %s эпох без улучшения (min_delta=%s). Остановка на эпохе %s.",
                    epochs_without_improvement,
                    args.early_stopping_min_delta,
                    epoch + 1,
                )
                break

        if args.metrics_json_path:
            metrics_path = Path(args.metrics_json_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            total_train_time_sec = float(time.perf_counter() - train_start_ts)
            train_dataset_size = int(len(train_loader.dataset))
            train_items_per_epoch = int(len(train_loader) * args.batch_size)
            effective_drop_ratio = (
                float(max(0, train_dataset_size - train_items_per_epoch) / train_dataset_size)
                if train_dataset_size > 0
                else 0.0
            )
            best_epoch = 0
            if epoch_stats:
                if args.task == "segmentation":
                    best_epoch = max(epoch_stats, key=lambda r: r["val_miou"])["epoch"]
                else:
                    best_epoch = max(epoch_stats, key=lambda r: r["val_accuracy"])["epoch"]
            # Сохраняем веса классов и названия классов — это критично для
            # интерпретации per_class_iou в свипе class_balance_beta.
            idx_to_class_map = {
                int(k): str(v) for k, v in getattr(train_dataset, "idx_to_class", {}).items()
            }
            payload = {
                "run_name": run_name,
                "model": args.model,
                "task": args.task,
                "dataset": args.dataset,
                "epochs": args.epochs,
                "epochs_completed": len(epoch_stats),
                "stopped_early": len(epoch_stats) < max(0, args.epochs - start_epoch),
                "batch_size": args.batch_size,
                "num_points": args.num_points,
                "amp": use_amp,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "class_balance": args.class_balance,
                "class_balance_beta": args.class_balance_beta,
                "lr_scheduler": args.lr_scheduler,
                "class_weights": class_weights_list,
                "idx_to_class": idx_to_class_map,
                "train_drop_last": bool(getattr(train_loader, "drop_last", False)),
                "train_dataset_size": train_dataset_size,
                "train_items_per_epoch": train_items_per_epoch,
                "train_steps_per_epoch": int(len(train_loader)),
                "effective_drop_ratio": effective_drop_ratio,
                "total_train_time_sec": total_train_time_sec,
                "best_val_metric": float(best_val_metric),
                "best_epoch": int(best_epoch),
                "max_peak_vram_mb": float(max((r["peak_vram_mb"] for r in epoch_stats), default=0.0)),
                "mean_epoch_time_sec": float(np.mean([r["epoch_time_sec"] for r in epoch_stats])) if epoch_stats else 0.0,
                "final_train_loss": float(epoch_stats[-1]["train_loss"]) if epoch_stats else None,
                "final_val_loss": float(epoch_stats[-1]["val_loss"]) if epoch_stats else None,
                "final_train_accuracy": float(epoch_stats[-1]["train_accuracy"]) if epoch_stats else None,
                "final_val_accuracy": float(epoch_stats[-1]["val_accuracy"]) if epoch_stats else None,
                "epoch_stats": epoch_stats,
            }
            if args.task == "segmentation":
                payload["best_val_miou"] = float(best_val_metric)
                payload["final_train_miou"] = float(epoch_stats[-1]["train_miou"]) if epoch_stats else None
                payload["final_val_miou"] = float(epoch_stats[-1]["val_miou"]) if epoch_stats else None
                payload["final_train_per_class_iou"] = (
                    list(epoch_stats[-1].get("train_per_class_iou", [])) if epoch_stats else None
                )
                payload["final_val_per_class_iou"] = (
                    list(epoch_stats[-1].get("val_per_class_iou", [])) if epoch_stats else None
                )
                payload["best_val_per_class_iou"] = (
                    list(best_val_per_class_iou) if best_val_per_class_iou is not None else None
                )
            else:
                payload["best_val_accuracy"] = float(best_val_metric)

            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("Сохранены метрики запуска: %s", metrics_path)


if __name__ == "__main__":
    main()
