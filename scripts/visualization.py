"""
Визуализация облаков точек через Open3D.

Режимы раскраски:
- pred: по предсказанным классам;
- gt: по ground-truth классам;
- rgb: по исходным цветам точек.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import open3d as o3d
except Exception as exc:
    raise RuntimeError("Open3D не установлен. Установите: pip install open3d") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import load_dataframe as _load_dataframe


def _load_predictions(predictions_path):
    if not predictions_path:
        return None
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Файл предсказаний не найден: {predictions_path}")
    return pd.read_csv(predictions_path, sep="\t")


def visualize(
    data_path,
    predictions_path=None,
    color_by="pred",
    max_points=None,
    seed=42,
    center=True,
):
    df = _load_dataframe(data_path)

    # Downsampling нужен для интерактивной скорости на больших сценах.
    if max_points is not None and len(df) > max_points:
        df = df.sample(n=max_points, random_state=seed)

    pred_df = _load_predictions(predictions_path) if predictions_path else None
    if pred_df is not None and max_points is not None and len(pred_df) > max_points:
        pred_df = pred_df.loc[df.index]

    if color_by == "pred":
        if pred_df is None or "Predicted_Classification" not in pred_df.columns:
            raise ValueError("Нет Predicted_Classification для режима 'pred'")
        labels = pred_df["Predicted_Classification"].to_numpy(dtype=np.int32)
        num_classes = int(labels.max()) + 1
        # Фиксируем seed для воспроизводимой палитры классов.
        palette = np.random.RandomState(seed).rand(num_classes, 3)
        colors = palette[labels]

    elif color_by == "gt":
        if "Classification" not in df.columns:
            raise ValueError("В файле нет Classification для режима 'gt'")
        labels = df["Classification"].to_numpy(dtype=np.int32)
        num_classes = int(labels.max()) + 1
        palette = np.random.RandomState(seed).rand(num_classes, 3)
        colors = palette[labels]

    elif color_by == "rgb":
        if not all(col in df.columns for col in ("R", "G", "B")):
            raise ValueError("В файле нет RGB колонок для режима 'rgb'")
        colors = df[["R", "G", "B"]].to_numpy(dtype=np.float32)
        max_val = float(colors.max()) if colors.size else 1.0
        if max_val > 255.0:
            colors = colors / 65535.0
        elif max_val > 1.0:
            colors = colors / 255.0

    else:
        raise ValueError("color_by должен быть 'pred', 'gt' или 'rgb'")

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if center:
        pcd.translate(-pcd.get_center())

    o3d.visualization.draw_geometries([pcd], width=900, height=700)


def main():
    parser = argparse.ArgumentParser(description="Визуализация point cloud через Open3D")
    parser.add_argument("--data", type=str, required=True, help="Путь к .txt/.las/.laz")
    parser.add_argument("--predictions", type=str, default=None, help="Файл с Predicted_Classification")
    parser.add_argument("--color_by", type=str, default="pred", choices=["pred", "gt", "rgb"])
    parser.add_argument("--max_points", type=int, default=None, help="Ограничить число точек")
    parser.add_argument("--seed", type=int, default=42, help="Сид для выборки и палитры")
    parser.add_argument("--no_center", action="store_true", help="Не центрировать облако")
    args = parser.parse_args()

    visualize(
        data_path=args.data,
        predictions_path=args.predictions,
        color_by=args.color_by,
        max_points=args.max_points,
        seed=args.seed,
        center=not args.no_center,
    )


if __name__ == "__main__":
    main()
