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
import os
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import amp
from contextlib import contextmanager
from sklearn.metrics import accuracy_score, confusion_matrix
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
)


def calculate_metrics(predictions, targets, num_classes, task="segmentation"):
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    if task == "classification":
        accuracy = accuracy_score(targets, predictions)
        cm = confusion_matrix(targets, predictions, labels=list(range(num_classes)))
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }

    # segmentation: считаем метрики на уровне отдельных точек.
    accuracy = accuracy_score(targets.flatten(), predictions.flatten())
    cm = confusion_matrix(targets.flatten(), predictions.flatten(), labels=list(range(num_classes)))

    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(0.0)

    mean_iou = float(np.mean(ious))
    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "per_class_iou": ious,
        "confusion_matrix": cm,
    }


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
                ce_per_sample = F.cross_entropy(
                    predictions,
                    labels,
                    weight=class_weights,
                    reduction="none",
                )
                if loss_type == "ce":
                    ce_loss = ce_per_sample.mean()
                elif loss_type in ("focal", "cb_focal"):
                    pt = torch.exp(-ce_per_sample)
                    ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                else:
                    raise ValueError(f"Неизвестный loss_type: {loss_type}")
                loss = ce_loss
                reg_loss = torch.tensor(0.0, device=predictions.device)
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                if model_type == "pointnet":
                    predictions, transform_coords, transform_features = model(features)
                    bsz, n_points, n_classes = predictions.shape
                    predictions_flat = predictions.reshape(-1, n_classes)
                    labels_flat = labels.reshape(-1)
                    ce_per_sample = F.cross_entropy(
                        predictions_flat,
                        labels_flat,
                        weight=class_weights,
                        reduction="none",
                    )
                    if loss_type == "ce":
                        ce_loss = ce_per_sample.mean()
                    elif loss_type in ("focal", "cb_focal"):
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type}")

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
                    predictions = model(features)
                    bsz, n_points, n_classes = predictions.shape
                    predictions_flat = predictions.reshape(-1, n_classes)
                    labels_flat = labels.reshape(-1)
                    ce_per_sample = F.cross_entropy(
                        predictions_flat,
                        labels_flat,
                        weight=class_weights,
                        reduction="none",
                    )
                    if loss_type == "ce":
                        ce_loss = ce_per_sample.mean()
                    elif loss_type in ("focal", "cb_focal"):
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type}")
                    loss = ce_loss
                    reg_loss = torch.tensor(0.0, device=predictions.device)
                pred_classes = torch.argmax(predictions, dim=2)

        optimizer.zero_grad()
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
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
):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for features, labels in pbar:
            features = features.float().to(device)
            labels = labels.long().to(device)

            with autocast_context(use_amp):
                if task == "classification":
                    predictions = model(features)
                    ce_per_sample = F.cross_entropy(
                        predictions,
                        labels,
                        weight=class_weights,
                        reduction="none",
                    )
                    if loss_type == "ce":
                        ce_loss = ce_per_sample.mean()
                    elif loss_type in ("focal", "cb_focal"):
                        pt = torch.exp(-ce_per_sample)
                        ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                    else:
                        raise ValueError(f"Неизвестный loss_type: {loss_type}")
                    loss = ce_loss
                    reg_loss = torch.tensor(0.0, device=predictions.device)
                    pred_classes = torch.argmax(predictions, dim=1)
                else:
                    if model_type == "pointnet":
                        predictions, transform_coords, transform_features = model(features)
                        bsz, n_points, n_classes = predictions.shape
                        predictions_flat = predictions.reshape(-1, n_classes)
                        labels_flat = labels.reshape(-1)
                        ce_per_sample = F.cross_entropy(
                            predictions_flat,
                            labels_flat,
                            weight=class_weights,
                            reduction="none",
                        )
                        if loss_type == "ce":
                            ce_loss = ce_per_sample.mean()
                        elif loss_type in ("focal", "cb_focal"):
                            pt = torch.exp(-ce_per_sample)
                            ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                        else:
                            raise ValueError(f"Неизвестный loss_type: {loss_type}")

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
                        predictions = model(features)
                        bsz, n_points, n_classes = predictions.shape
                        predictions_flat = predictions.reshape(-1, n_classes)
                        labels_flat = labels.reshape(-1)
                        ce_per_sample = F.cross_entropy(
                            predictions_flat,
                            labels_flat,
                            weight=class_weights,
                            reduction="none",
                        )
                        if loss_type == "ce":
                            ce_loss = ce_per_sample.mean()
                        elif loss_type in ("focal", "cb_focal"):
                            pt = torch.exp(-ce_per_sample)
                            ce_loss = (((1.0 - pt) ** focal_gamma) * ce_per_sample).mean()
                        else:
                            raise ValueError(f"Неизвестный loss_type: {loss_type}")
                        loss = ce_loss
                        reg_loss = torch.tensor(0.0, device=predictions.device)
                    pred_classes = torch.argmax(predictions, dim=2)

            total_loss += loss.item()
            all_predictions.append(pred_classes)
            all_targets.append(labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

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


def build_model(
    model_type,
    task,
    num_classes,
    num_features,
    k=20,
    k_small=20,
    k_large=40,
    attention_type="none",
    attention_k=16,
    attention_heads=4,
    attention_dropout=0.1,
):
    if model_type == "pointnet":
        return (
            PointNetSegmentation(num_classes=num_classes, num_features=num_features)
            if task == "segmentation"
            else PointNetClassification(num_classes=num_classes, num_features=num_features)
        )
    if model_type == "pointnet++":
        return (
            PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
            if task == "segmentation"
            else PointNetPlusPlusClassification(num_classes=num_classes, num_features=num_features)
        )
    if model_type == "dgcnn":
        return (
            DGCNNSegmentation(num_classes=num_classes, num_features=num_features, k=k)
            if task == "segmentation"
            else DGCNNClassification(num_classes=num_classes, num_features=num_features, k=k)
        )
    if model_type == "ldgcnn":
        return (
            LDGCNNSegmentation(
                num_classes=num_classes,
                num_features=num_features,
                k_small=k_small,
                k_large=k_large,
                attention_type=attention_type,
                attention_k=attention_k,
                attention_heads=attention_heads,
                attention_dropout=attention_dropout,
            )
            if task == "segmentation"
            else LDGCNNClassification(
                num_classes=num_classes,
                num_features=num_features,
                k_small=k_small,
                k_large=k_large,
                attention_type=attention_type,
                attention_k=attention_k,
                attention_heads=attention_heads,
                attention_dropout=attention_dropout,
            )
        )
    raise ValueError(f"Неизвестная модель: {model_type}")


def set_seed(seed):
    """
    Фиксирует источники случайности для воспроизводимых запусков.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Для воспроизводимости отключаем недетерминированные fast-path режимы.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Для совместимости со старыми версиями PyTorch без warn_only.
        torch.use_deterministic_algorithms(True)


def make_worker_init_fn(base_seed):
    """
    Инициализирует RNG каждого DataLoader worker своим seed.
    """

    def seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


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
    parser.add_argument("--num_classes", type=int, default=None, help="Количество классов")
    parser.add_argument("--lambda_reg", type=float, default=0.001, help="Коэффициент регуляризации трансформаций")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Директория для сохранения моделей")
    parser.add_argument("--resume", type=str, default=None, help="Путь к чекпоинту для возобновления обучения")
    parser.add_argument("--model", type=str, default="pointnet", choices=["pointnet", "pointnet++", "dgcnn", "ldgcnn"], help="Модель")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "classification"], help="Задача")
    parser.add_argument("--experiment_name", type=str, default="PointCloudExperiments", help="MLflow experiment name")
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
    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal", "cb_focal"], help="Тип функции потерь")
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

    args = parser.parse_args()
    if args.loss_type == "cb_focal" and args.class_balance == "none":
        raise ValueError("Для --loss_type cb_focal необходимо включить --class_balance (inverse/effective).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    if os.name == "nt" and args.num_workers > 0 and not args.allow_windows_workers:
        print("Windows: num_workers>0 может приводить к ошибкам. Устанавливаю num_workers=0.")
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
    print(f"Train data: {train_data}")
    print(f"Val data: {val_data}")

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
    )

    if args.cache_only:
        print("Кэш подготовлен. Завершение без обучения (--cache_only).")
        return

    num_classes = train_dataset.num_classes if args.num_classes is None else args.num_classes
    print(f"Количество классов: {num_classes}")
    class_weights = compute_class_weights(
        train_dataset,
        num_classes=num_classes,
        task=args.task,
        mode=args.class_balance,
        beta=args.class_balance_beta,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
        print(f"Включена балансировка классов: {args.class_balance}")
        print(f"Class weights: {class_weights.detach().cpu().numpy()}")

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True if torch.cuda.is_available() else False,
        "drop_last": True,
    }
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader_kwargs["worker_init_fn"] = make_worker_init_fn(args.seed)
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
        "drop_last": True,
    }
    val_loader_kwargs["worker_init_fn"] = make_worker_init_fn(args.seed + 10_000)
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
    print(f"Используется модель: {args.model} ({args.task})")
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    best_val_iou = 0

    if args.resume:
        print(f"Загрузка чекпоинта из {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        checkpoint_model_type = checkpoint.get("model_type", args.model)
        checkpoint_task = checkpoint.get("task", args.task)
        checkpoint_attention_type = checkpoint.get("attention_type", "none")
        if checkpoint_model_type != args.model:
            print(f"Предупреждение: модель в чекпоинте ({checkpoint_model_type}) не совпадает с аргументом ({args.model})")
        if checkpoint_task != args.task:
            print(f"Предупреждение: task в чекпоинте ({checkpoint_task}) не совпадает с аргументом ({args.task})")
        if checkpoint_attention_type != args.attention_type:
            print(
                "Предупреждение: attention_type в чекпоинте "
                f"({checkpoint_attention_type}) не совпадает с аргументом ({args.attention_type})"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_iou = checkpoint.get("best_val_iou", 0)

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_scaler(use_amp)

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{args.model}_{args.task}_{args.dataset}"):
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
                "num_classes": num_classes,
                "num_workers": args.num_workers,
                "seed": args.seed,
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
            }
        )

        print("Начало обучения")
        for epoch in range(start_epoch, args.epochs):
            print(f"\nЭпоха {epoch + 1}/{args.epochs}")
            print("-" * 50)

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
            )
            scheduler.step()

            print(
                f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                f"Accuracy: {train_metrics['accuracy']:.4f}"
                + (f", mIoU: {train_metrics['mean_iou']:.4f}" if args.task == "segmentation" else "")
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f}"
                + (f", mIoU: {val_metrics['mean_iou']:.4f}" if args.task == "segmentation" else "")
            )

            # MLflow metrics
            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric("train_accuracy", train_metrics["accuracy"], step=epoch)
            mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
            if args.task == "segmentation":
                mlflow.log_metric("train_miou", train_metrics["mean_iou"], step=epoch)
                mlflow.log_metric("val_miou", val_metrics["mean_iou"], step=epoch)

            # Сохраняем лучшую модель по целевой метрике:
            # mIoU для segmentation, accuracy для classification.
            if args.task == "segmentation":
                metric_for_best = val_metrics["mean_iou"]
            else:
                metric_for_best = val_metrics["accuracy"]

            if metric_for_best > best_val_iou:
                best_val_iou = metric_for_best
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_iou": best_val_iou,
                    "num_classes": num_classes,
                    "num_features": num_features,
                    "model_type": args.model,
                    "task": args.task,
                    "attention_type": args.attention_type,
                    "attention_k": args.attention_k,
                    "attention_heads": args.attention_heads,
                    "attention_dropout": args.attention_dropout,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                    "class_balance": args.class_balance,
                    "class_balance_beta": args.class_balance_beta,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }
                best_path = os.path.join(args.save_dir, "best_model.pth")
                torch.save(checkpoint, best_path)
                mlflow.log_artifact(best_path)
                print(f"Сохранена лучшая модель: {best_path}")

            # Сохранение последнего чекпоинта
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "num_classes": num_classes,
                "num_features": num_features,
                "model_type": args.model,
                "task": args.task,
                "attention_type": args.attention_type,
                "attention_k": args.attention_k,
                "attention_heads": args.attention_heads,
                "attention_dropout": args.attention_dropout,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
                "class_balance": args.class_balance,
                "class_balance_beta": args.class_balance_beta,
            }
            last_path = os.path.join(args.save_dir, "last_checkpoint.pth")
            torch.save(checkpoint, last_path)


if __name__ == "__main__":
    main()
