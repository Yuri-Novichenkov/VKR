"""
Визуализация облаков точек.

Режимы раскраски:
- pred: по предсказанным классам;
- gt: по ground-truth классам;
- rgb: по исходным цветам точек.

Режимы вывода:
- интерактивный (Open3D, по умолчанию);
- сохранение PNG с легендой (--save_png путь).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.io import load_dataframe as _load_dataframe


def _load_class_names(config_path: str) -> list[str]:
    """Загружает имена классов из YAML-файла (configs/classes/*.yaml)."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    classes = data.get("classes", {})
    max_id = max(int(k) for k in classes)
    names = [""] * (max_id + 1)
    for k, v in classes.items():
        names[int(k)] = v
    return names


def _get_palette(num_classes: int) -> np.ndarray:
    """Фиксированная палитра tab20 (до 20 классов), RGB float [0,1]."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20", max(num_classes, 1))
    return np.array([cmap(i)[:3] for i in range(num_classes)])


def _load_predictions(predictions_path: str) -> pd.DataFrame:
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Файл предсказаний не найден: {predictions_path}")
    return pd.read_csv(predictions_path, sep="\t")


def _resolve_labels(df: pd.DataFrame, pred_df: pd.DataFrame | None,
                    color_by: str, num_classes: int | None):
    """Возвращает (labels np.int32, palette np.ndarray(N,3))."""
    if color_by == "pred":
        if pred_df is None or "Predicted_Classification" not in pred_df.columns:
            raise ValueError("Нет Predicted_Classification для режима 'pred'")
        labels = pred_df["Predicted_Classification"].to_numpy(dtype=np.int32)
    elif color_by == "gt":
        if "Classification" not in df.columns:
            raise ValueError("В файле нет колонки Classification для режима 'gt'")
        labels = df["Classification"].to_numpy(dtype=np.int32)
    else:
        raise ValueError(f"color_by={color_by!r} не поддерживается (нужен 'pred' или 'gt')")

    n = num_classes if num_classes is not None else int(labels.max()) + 1
    palette = _get_palette(n)
    return labels, palette


def _print_legend(labels: np.ndarray, palette: np.ndarray,
                  class_names: list[str] | None):
    print("\nЛегенда классов:")
    for cls_id in sorted(np.unique(labels).tolist()):
        name = (class_names[cls_id]
                if class_names and cls_id < len(class_names)
                else f"Класс {cls_id}")
        rgb = (palette[cls_id] * 255).astype(int)
        print(f"  [{cls_id:2d}] {name:<30s}  RGB=({rgb[0]:3d},{rgb[1]:3d},{rgb[2]:3d})")


def _save_legend_png(labels: np.ndarray, palette: np.ndarray,
                     class_names: list[str] | None,
                     output_path: str = "figures/legend.png"):
    """Сохраняет легенду как PNG — без открытия GUI (нет конфликта с Open3D)."""
    import matplotlib
    matplotlib.use("Agg")  # без GUI backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams["font.family"] = "DejaVu Sans"  # поддерживает кириллицу

    present = sorted(np.unique(labels).tolist())
    patches = []
    for cls_id in present:
        name = (class_names[cls_id]
                if class_names and cls_id < len(class_names)
                else f"Класс {cls_id}")
        patches.append(mpatches.Patch(color=palette[cls_id],
                                      label=f"[{cls_id}]  {name}"))

    fig, ax = plt.subplots(figsize=(3.5, 0.45 * len(present) + 0.5))
    ax.axis("off")
    ax.legend(handles=patches, loc="center", fontsize=11,
              framealpha=0.0, handlelength=1.5, handleheight=1.2)
    fig.suptitle("Легенда классов", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Легенда сохранена: {output_path}")


# ── Open3D (интерактивный режим) ──────────────────────────────────────────────

def visualize_open3d(
    df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    color_by: str,
    num_classes: int | None,
    class_names: list[str] | None,
    center: bool = True,
    legend: bool = False,
):
    try:
        import open3d as o3d
    except ImportError:
        raise RuntimeError("Open3D не установлен: pip install open3d")

    if color_by == "rgb":
        raw = df[["R", "G", "B"]].to_numpy(dtype=np.float32)
        max_val = float(raw.max()) if raw.size else 1.0
        if max_val > 255.0:
            raw /= 65535.0
        elif max_val > 1.0:
            raw /= 255.0
        colors = raw.astype(np.float64)
        labels, palette = None, None
    else:
        labels, palette = _resolve_labels(df, pred_df, color_by, num_classes)
        colors = palette[labels % len(palette)]

    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    if center:
        pcd.translate(-pcd.get_center())

    if labels is not None and palette is not None:
        _print_legend(labels, palette, class_names)
        if legend:
            _save_legend_png(labels, palette, class_names)

    o3d.visualization.draw_geometries([pcd], width=900, height=700)


# ── Matplotlib (сохранение PNG) ───────────────────────────────────────────────

def save_png(
    df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
    color_by: str,
    num_classes: int | None,
    class_names: list[str] | None,
    output_path: str,
    max_points: int = 300_000,
    title: str | None = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Сэмплинг для скорости matplotlib
    if len(df) > max_points:
        rng = np.random.RandomState(42)
        idx = np.sort(rng.choice(len(df), max_points, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        if pred_df is not None:
            pred_df = pred_df.iloc[idx].reset_index(drop=True)

    labels, palette = _resolve_labels(df, pred_df, color_by, num_classes)
    colors = palette[labels % len(palette)]

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, c=colors, s=0.5, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xlabel("X (м)", fontsize=11)
    ax.set_ylabel("Y (м)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=13)

    # Легенда по присутствующим классам
    present = sorted(np.unique(labels).tolist())
    patches = []
    for cls_id in present:
        name = (class_names[cls_id]
                if class_names and cls_id < len(class_names)
                else f"Класс {cls_id}")
        patches.append(mpatches.Patch(color=palette[cls_id], label=f"[{cls_id}] {name}"))
    ax.legend(handles=patches, loc="upper right", fontsize=9,
              framealpha=0.85, title="Классы", title_fontsize=10)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Визуализация point cloud (Open3D интерактив или PNG с легендой)"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Путь к .txt/.las/.laz")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Файл с Predicted_Classification (для --color_by pred)")
    parser.add_argument("--color_by", type=str, default="pred",
                        choices=["pred", "gt", "rgb"],
                        help="Способ раскраски (default: pred)")
    parser.add_argument("--max_points", type=int, default=None,
                        help="Максимум точек (default: все)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Сид для сэмплинга (default: 42)")
    parser.add_argument("--no_center", action="store_true",
                        help="Не центрировать облако (только Open3D)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Число классов для фиксированной палитры")
    parser.add_argument("--class_names", nargs="+", default=None,
                        help="Имена классов 0,1,2,... Пример: --class_names Ground Vegetation Building")
    parser.add_argument("--class_config", type=str, default=None,
                        help="YAML-файл с именами классов (configs/classes/hessigheim.yaml)")
    parser.add_argument("--legend", action="store_true",
                        help="Показать легенду в отдельном окне рядом с Open3D")
    parser.add_argument("--save_png", type=str, default=None,
                        help="Путь для сохранения PNG (вместо Open3D)")
    parser.add_argument("--title", type=str, default=None,
                        help="Заголовок графика (только для --save_png)")
    args = parser.parse_args()

    # Загрузка имён классов: --class_config имеет приоритет над --class_names
    if args.class_config:
        args.class_names = _load_class_names(args.class_config)

    df = _load_dataframe(args.data)
    pred_df = _load_predictions(args.predictions) if args.predictions else None

    # Для Open3D сэмплируем заранее
    if args.save_png is None and args.max_points and len(df) > args.max_points:
        df = df.sample(n=args.max_points, random_state=args.seed)
        if pred_df is not None:
            pred_df = pred_df.loc[df.index]

    if args.save_png:
        save_png(
            df, pred_df,
            color_by=args.color_by,
            num_classes=args.num_classes,
            class_names=args.class_names,
            output_path=args.save_png,
            max_points=args.max_points or 300_000,
            title=args.title,
        )
    else:
        visualize_open3d(
            df, pred_df,
            color_by=args.color_by,
            num_classes=args.num_classes,
            class_names=args.class_names,
            center=not args.no_center,
            legend=args.legend,
        )


if __name__ == "__main__":
    main()
