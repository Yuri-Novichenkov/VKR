import argparse
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
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
        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
        }

    # segmentation
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
                loss, ce_loss, reg_loss = model.get_loss(predictions, labels)
                pred_classes = torch.argmax(predictions, dim=1)
            else:
                if model_type == "pointnet":
                    predictions, transform_coords, transform_features = model(features)
                    loss, ce_loss, reg_loss = model.get_loss(
                        predictions, labels, transform_coords, transform_features, lambda_reg
                    )
                else:
                    predictions = model(features)
                    loss, ce_loss, reg_loss = model.get_loss(predictions, labels)
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


def validate(model, val_loader, device, num_classes, model_type, task="segmentation", use_amp=False):
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
                    loss, ce_loss, reg_loss = model.get_loss(predictions, labels)
                    pred_classes = torch.argmax(predictions, dim=1)
                else:
                    if model_type == "pointnet":
                        predictions, transform_coords, transform_features = model(features)
                        loss, ce_loss, reg_loss = model.get_loss(
                            predictions, labels, transform_coords, transform_features
                        )
                    else:
                        predictions = model(features)
                        loss, ce_loss, reg_loss = model.get_loss(predictions, labels)
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

    data_root = args.data_root
    if data_root is None:
        data_root = os.path.join("Files", args.dataset, "LiDAR")

    train_data = os.path.join(data_root, f"{args.dataset}_train.txt")
    val_data = os.path.join(data_root, f"{args.dataset}_val.txt")
    return train_data, val_data


def build_model(model_type, task, num_classes, num_features, k=20, k_small=20, k_large=40):
    if model_type == "pointnet":
        return PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "pointnet++":
        return PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "dgcnn":
        return (
            DGCNNSegmentation(num_classes=num_classes, num_features=num_features, k=k)
            if task == "segmentation"
            else DGCNNClassification(num_classes=num_classes, num_features=num_features, k=k)
        )
    if model_type == "ldgcnn":
        return (
            LDGCNNSegmentation(num_classes=num_classes, num_features=num_features, k_small=k_small, k_large=k_large)
            if task == "segmentation"
            else LDGCNNClassification(num_classes=num_classes, num_features=num_features, k_small=k_small, k_large=k_large)
        )
    raise ValueError(f"Неизвестная модель: {model_type}")


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
    parser.add_argument("--cache_dir", type=str, default="cache", help="Директория для кэша npz")
    parser.add_argument("--cache_mode", type=str, default="write", choices=["off", "read", "write"], help="Режим кэша")
    parser.add_argument("--cache_chunked", action="store_true", help="Сохранять нарезанные облака чанками")
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка по облакам")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="prefetch_factor для DataLoader")
    parser.add_argument("--persistent_workers", action="store_true", help="persistent_workers для DataLoader")
    parser.add_argument("--cache_only", action="store_true", help="Только подготовить кэш и выйти")
    parser.add_argument("--allow_windows_workers", action="store_true", help="Разрешить num_workers>0 на Windows")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if os.name == "nt" and args.num_workers > 0 and not args.allow_windows_workers:
        print("Windows: num_workers>0 может приводить к ошибкам. Устанавливаю num_workers=0.")
        args.num_workers = 0

    if args.save_dir == "checkpoints":
        args.save_dir = os.path.join(args.save_dir, args.model, args.dataset.lower())
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

    train_loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True if torch.cuda.is_available() else False,
        "drop_last": True,
    }
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader_kwargs["persistent_workers"] = args.persistent_workers

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
        if checkpoint_model_type != args.model:
            print(f"Предупреждение: модель в чекпоинте ({checkpoint_model_type}) не совпадает с аргументом ({args.model})")
        if checkpoint_task != args.task:
            print(f"Предупреждение: task в чекпоинте ({checkpoint_task}) не совпадает с аргументом ({args.task})")
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
                "amp": use_amp,
                "k": args.k,
                "k_small": args.k_small,
                "k_large": args.k_large,
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
            )
            val_metrics = validate(model, val_loader, device, num_classes, args.model, task=args.task, use_amp=use_amp)
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

            # Сохранение лучшей модели
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
            }
            last_path = os.path.join(args.save_dir, "last_checkpoint.pth")
            torch.save(checkpoint, last_path)


if __name__ == "__main__":
    main()
