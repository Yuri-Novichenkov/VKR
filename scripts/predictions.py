"""
Генерация предсказаний сегментации и сохранение их в tab-separated файл.

Важно: скрипт предназначен именно для task=segmentation
и добавляет колонку Predicted_Classification.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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


def load_dataframe(data_path):
    # Функция дублирует логику чтения из Dataset, чтобы сохранить
    # исходную таблицу с теми же колонками + предсказания.
    path = Path(data_path)
    suffix = path.suffix.lower()
    if suffix in (".laz", ".las"):
        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "Для чтения .laz/.las установите laspy и lazrs: "
                "pip install laspy lazrs"
            ) from exc

        las = laspy.read(str(path))
        data = {
            "X": np.asarray(las.x),
            "Y": np.asarray(las.y),
            "Z": np.asarray(las.z),
        }
        if hasattr(las, "red"):
            data["R"] = np.asarray(las.red)
        if hasattr(las, "green"):
            data["G"] = np.asarray(las.green)
        if hasattr(las, "blue"):
            data["B"] = np.asarray(las.blue)
        if hasattr(las, "intensity"):
            data["Intensity"] = np.asarray(las.intensity)
        if hasattr(las, "number_of_returns"):
            data["NumberOfReturns"] = np.asarray(las.number_of_returns)
        if hasattr(las, "return_number"):
            data["ReturnNumber"] = np.asarray(las.return_number)
        if hasattr(las, "classification"):
            data["Classification"] = np.asarray(las.classification)
        return pd.DataFrame(data)

    return pd.read_csv(path, sep="\t")


def main():
    parser = argparse.ArgumentParser(description="Генерация предсказаний модели")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--data", type=str, required=True, help="Файл данных (.txt/.las/.laz)")
    parser.add_argument("--output_root", type=str, default="predictions", help="Папка для предсказаний")
    parser.add_argument("--num_points", type=int, default=4096, help="Количество точек в облаке")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Устройство")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Директория для кэша")
    parser.add_argument("--cache_mode", type=str, default="read", choices=["off", "read", "write"])
    parser.add_argument("--cache_chunked", action="store_true", help="Использовать кэш чанков")
    parser.add_argument("--chunk_size", type=int, default=512, help="Размер чанка по облакам")
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

    # Загружаем архитектурные параметры из checkpoint,
    # чтобы корректно восстановить модель.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    num_features = checkpoint["num_features"]
    model_type = checkpoint.get("model_type", "pointnet")
    task = checkpoint.get("task", "segmentation")
    if task != "segmentation":
        raise ValueError("predictions.py поддерживает только task=segmentation")

    model = build_model(model_type, num_classes, num_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    dataset = LiDARDataset(
        args.data,
        num_points=args.num_points,
        augment=False,
        has_labels=False,
        task="segmentation",
        cache_dir=args.cache_dir,
        cache_mode=args.cache_mode,
        cache_chunked=args.cache_chunked,
        chunk_size=args.chunk_size,
    )
    dataset.num_classes = num_classes

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_predictions = []
    with torch.no_grad():
        for features, _labels in loader:
            features = features.float().to(device)
            if model_type == "pointnet":
                predictions, _, _ = model(features)
            else:
                predictions = model(features)
            pred_classes = torch.argmax(predictions, dim=2)
            all_predictions.append(pred_classes.cpu().numpy())

    # После batch-предсказаний разворачиваем в единый вектор.
    all_predictions = np.concatenate(all_predictions, axis=0).reshape(-1)

    df = load_dataframe(args.data)
    if len(all_predictions) > len(df):
        all_predictions = all_predictions[: len(df)]

    df = df.copy()
    df["Predicted_Classification"] = all_predictions[: len(df)]

    data_stem = Path(args.data).stem
    output_dir = Path(args.output_root) / model_type / data_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{data_stem}_predictions.txt"
    df.to_csv(output_path, sep="\t", index=False)
    print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    main()
