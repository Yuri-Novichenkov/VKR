"""
Build comparative plots for sweep runs from runs_summary.csv.

Два режима (автоопределение по колонкам CSV):

**single-axis** (batch_size / class_balance_beta)
  Старый режим: одна модель, числовая ось X. Графики: line plots метрик,
  per-class IoU, hard-classes IoU.

**matrix** (models × loss_configs)
  Новый режим: несколько моделей × несколько конфигов потерь.
  Графики:
    1. Grouped bar — mIoU по loss_config, группировка по модели
    2. Grouped bar — mIoU по модели, группировка по loss_config
    3. Heatmap — model × loss_config → best_val_mIoU
    4. Grouped bar — accuracy по loss_config
    5. Grouped bar — epoch time по loss_config
    6. Per-class IoU heatmap — для каждой модели (из JSON-метрик)
    7. Radar chart — mIoU по классам для лучшей модели × loss_config
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


# ─────────────────────────── общие утилиты ────────────────────────────────

_AXIS_FILE_PREFIX = {
    "batch_size": "batch",
    "class_balance_beta": "beta",
}
_AXIS_XLABEL = {
    "batch_size": "batch_size",
    "class_balance_beta": "class_balance_beta",
}

# Человекочитаемые имена конфигов потерь (если есть в данных)
_LOSS_DISPLAY = {
    "ce_baseline":  "CE",
    "ce_cb":        "CE + CB",
    "focal":        "Focal",
    "focal_cb":     "Focal + CB",
    "lovasz_cb":    "Lovász + CB",
    # запасные варианты
    "ce":           "CE",
    "focal_gamma2": "Focal γ=2",
}


def _nice_loss_name(name: str) -> str:
    return _LOSS_DISPLAY.get(name, name)


def _detect_mode(df: pd.DataFrame) -> str:
    """matrix если есть колонки model и loss_config_name, иначе single."""
    if "model" in df.columns and "loss_config_name" in df.columns:
        return "matrix"
    return "single"


# ─────────────────────────── single-axis графики ──────────────────────────

def _safe_divide(num, den):
    out = num / den
    return out.replace([float("inf"), float("-inf")], pd.NA)


def _plot_line(df, x_col, y_col, out_path, title, ylabel):
    if y_col not in df.columns:
        return
    part = df[[x_col, y_col]].dropna().sort_values(x_col)
    if part.empty:
        return
    use_cat = x_col == "class_balance_beta"
    if use_cat:
        x_values = list(range(len(part)))
        x_labels = [f"{float(v):.5g}" for v in part[x_col].tolist()]
    else:
        x_values = part[x_col].tolist()
        x_labels = None
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, part[y_col], marker="o")
    for i, (_, row) in enumerate(part.iterrows()):
        plt.annotate(
            f"{row[y_col]:.3f}" if isinstance(row[y_col], float) else str(row[y_col]),
            (x_values[i], row[y_col]),
        )
    plt.title(title)
    plt.xlabel(_AXIS_XLABEL.get(x_col, x_col))
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if use_cat and x_labels:
        plt.xticks(x_values, x_labels, rotation=25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_per_class_iou_single(df, x_col, output_dir, prefix, summary_dir):
    if "metrics_path" not in df.columns:
        return
    records_best, records_final = [], []
    for _, row in df.iterrows():
        mp = row.get("metrics_path")
        if not isinstance(mp, str) or not mp.strip():
            continue
        mp_obj = Path(mp)
        candidates = [mp_obj, summary_dir / "metrics" / mp_obj.name]
        metrics = None
        for c in candidates:
            try:
                metrics = json.loads(c.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
        if not metrics:
            continue
        idx_to_class = metrics.get("idx_to_class", {})
        x_val = row.get(x_col)
        for tag, key, records in [
            ("best", "best_val_per_class_iou", records_best),
            ("final", "final_val_per_class_iou", records_final),
        ]:
            lst = metrics.get(key)
            if not isinstance(lst, list):
                continue
            for idx, iou in enumerate(lst):
                if iou is None:
                    continue
                records.append({
                    "x": x_val, "class_idx": idx,
                    "class_name": str(idx_to_class.get(str(idx), idx)),
                    "iou": float(iou),
                })

    def _plot(records, filename, title):
        if not records:
            return
        part = pd.DataFrame(records).dropna(subset=["x", "iou"]).sort_values("x")
        if part.empty:
            return
        use_cat = x_col == "class_balance_beta"
        if use_cat:
            unique_x = sorted(part["x"].dropna().unique())
            x_to_pos = {x: i for i, x in enumerate(unique_x)}
            part["x_plot"] = part["x"].map(x_to_pos)
            xticks = list(range(len(unique_x)))
            xlabels = [f"{float(v):.5g}" for v in unique_x]
        else:
            part["x_plot"] = part["x"]
            xticks = xlabels = None
        plt.figure(figsize=(11, 7))
        for ci in sorted(part["class_idx"].unique()):
            cp = part[part["class_idx"] == ci].sort_values("x")
            label = f"class {int(ci)} ({cp['class_name'].iloc[0]})"
            plt.plot(cp["x_plot"], cp["iou"], marker="o", linewidth=1.5, label=label)
        plt.title(title)
        plt.xlabel(_AXIS_XLABEL.get(x_col, x_col))
        plt.ylabel("IoU")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        if use_cat and xticks is not None:
            plt.xticks(xticks, xlabels, rotation=25)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    _plot(records_best, f"{prefix}_vs_best_val_per_class_iou.png",
          f"{_AXIS_XLABEL.get(x_col, x_col)} vs best val per-class IoU")
    _plot(records_final, f"{prefix}_vs_final_val_per_class_iou.png",
          f"{_AXIS_XLABEL.get(x_col, x_col)} vs final val per-class IoU")

    if records_best:
        best_df = pd.DataFrame(records_best).dropna(subset=["x", "iou"])
        if not best_df.empty:
            worst = (best_df.groupby("class_idx")["iou"].mean()
                     .sort_values().head(3).index.tolist())
            hard_df = best_df[best_df["class_idx"].isin(worst)].copy()
            if not hard_df.empty:
                agg = (hard_df.groupby("x", as_index=False)["iou"]
                       .mean().rename(columns={"iou": "hard_classes_mean_iou"}))
                _plot_line(
                    agg.rename(columns={"x": x_col}), x_col, "hard_classes_mean_iou",
                    output_dir / f"{prefix}_vs_hard_classes_mean_iou.png",
                    f"{_AXIS_XLABEL.get(x_col, x_col)} vs hard-classes mean IoU",
                    "hard_classes_mean_iou",
                )


def _detect_x_col(df):
    candidates = ["class_balance_beta", "batch_size"]
    varying = [c for c in candidates if c in df.columns and df[c].dropna().nunique() > 1]
    return varying[0] if varying else "batch_size"


def plot_single_axis(df, summary_path, output_dir):
    if "return_code" in df.columns:
        ok = df[df["return_code"] == 0].copy()
    else:
        ok = df.copy()
    if ok.empty:
        print("No successful runs.")
        return

    x_col = _detect_x_col(ok)
    prefix = _AXIS_FILE_PREFIX.get(x_col, x_col)
    xlabel = _AXIS_XLABEL.get(x_col, x_col)

    _plot_line(ok, x_col, "best_val_metric",       output_dir / f"{prefix}_vs_best_val_metric.png",       f"{xlabel} vs best val metric",       "best_val_metric")
    _plot_line(ok, x_col, "final_val_miou",         output_dir / f"{prefix}_vs_final_val_miou.png",         f"{xlabel} vs final val mIoU",         "final_val_miou")
    _plot_line(ok, x_col, "final_val_accuracy",     output_dir / f"{prefix}_vs_final_val_accuracy.png",     f"{xlabel} vs final val accuracy",     "final_val_accuracy")
    _plot_line(ok, x_col, "mean_epoch_time_sec",    output_dir / f"{prefix}_vs_mean_epoch_time_sec.png",    f"{xlabel} vs mean epoch time",        "mean_epoch_time_sec")
    _plot_line(ok, x_col, "total_train_time_sec",   output_dir / f"{prefix}_vs_total_train_time_sec.png",   f"{xlabel} vs total train time",       "total_train_time_sec")
    _plot_line(ok, x_col, "max_peak_vram_mb",       output_dir / f"{prefix}_vs_max_peak_vram_mb.png",       f"{xlabel} vs peak VRAM",              "max_peak_vram_mb")

    if {"mean_epoch_time_sec", "train_items_per_epoch"}.issubset(ok.columns):
        p = ok[[x_col, "mean_epoch_time_sec", "train_items_per_epoch"]].dropna().copy()
        p["samples_per_sec"] = _safe_divide(p["train_items_per_epoch"], p["mean_epoch_time_sec"])
        _plot_line(p, x_col, "samples_per_sec",
                   output_dir / f"{prefix}_vs_samples_per_sec.png",
                   f"{xlabel} vs samples per second", "samples_per_sec")
    if {"mean_epoch_time_sec", "train_items_per_epoch", "num_points"}.issubset(ok.columns):
        p = ok[[x_col, "mean_epoch_time_sec", "train_items_per_epoch", "num_points"]].dropna().copy()
        p["points_per_sec"] = _safe_divide(
            p["train_items_per_epoch"] * p["num_points"], p["mean_epoch_time_sec"])
        _plot_line(p, x_col, "points_per_sec",
                   output_dir / f"{prefix}_vs_points_per_sec.png",
                   f"{xlabel} vs points per second", "points_per_sec")

    _plot_per_class_iou_single(ok, x_col, output_dir, prefix, summary_path.parent)
    print(f"Single-axis plots saved to: {output_dir}")


# ─────────────────────────── matrix графики ───────────────────────────────

# Порядок конфигов потерь для оси X (если есть в данных)
_LOSS_ORDER = ["ce_baseline", "ce_cb", "focal", "focal_cb", "lovasz_cb"]

# Семантические имена LAS-классов (стандарт ASPRS)
_LAS_CLASS_NAMES = {
    0:  "Unclassified",
    1:  "Ground",
    2:  "Low Veg.",
    3:  "Med. Veg.",
    4:  "High Veg.",
    5:  "Building",
    6:  "Noise",
    7:  "Key-point",
    8:  "Water",
    9:  "Reserved-9",
    10: "Reserved-10",
}


def _get_class_label(ci: int, idx_to_class: dict) -> str:
    """Возвращает читаемое имя класса. Приоритет: idx_to_class → LAS → 'cls{ci}'."""
    raw = idx_to_class.get(str(ci), idx_to_class.get(ci))
    # Если raw совпадает с числовым индексом — это пустой маппинг, используем LAS
    try:
        if raw is not None and int(str(raw)) == ci:
            return _LAS_CLASS_NAMES.get(ci, f"cls{ci}")
    except (ValueError, TypeError):
        if raw is not None:
            return str(raw)
    return _LAS_CLASS_NAMES.get(ci, f"cls{ci}")


def _simple_bar(series, out_path, title, xlabel, ylabel, fmt=".3f", figsize=(10, 5), ylim=None):
    """Простой bar chart: X = категории (loss configs), одна серия значений."""
    fig, ax = plt.subplots(figsize=figsize)
    names = [_nice_loss_name(n) for n in series.index]
    vals = series.values.astype(float)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(vals)))
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved: {out_path.name}")


def _loss_order_key(name):
    try:
        return _LOSS_ORDER.index(name)
    except ValueError:
        return 999


def _grouped_bar(
    pivot: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 20,
    figsize=(12, 6),
    ylim=None,
    fmt=".3f",
    annot=True,
):
    """Grouped bar chart из pivot-таблицы (строки=группы, колонки=серии)."""
    models = list(pivot.index)
    loss_names = list(pivot.columns)
    n_groups = len(models)
    n_bars = len(loss_names)
    x = np.arange(n_groups)
    width = 0.8 / max(n_bars, 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_bars))

    fig, ax = plt.subplots(figsize=figsize)
    for i, lc in enumerate(loss_names):
        vals = pivot[lc].values.astype(float)
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=_nice_loss_name(lc), color=colors[i])
        if annot:
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:{fmt}}",
                        ha="center", va="bottom", fontsize=7, rotation=0,
                    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=rotation, ha="right", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved: {out_path.name}")


def _heatmap(
    pivot: pd.DataFrame,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    fmt=".3f",
    cmap="YlOrRd",
):
    """Тепловая карта из pivot (строки=models, колонки=loss_configs)."""
    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5),
                                    max(4, len(pivot.index) * 0.8)))
    data = pivot.values.astype(float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([_nice_loss_name(c) for c in pivot.columns], rotation=25, ha="right")
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = data[i, j]
            if not np.isnan(v):
                text_color = "white" if v < (vmin + (vmax - vmin) * 0.6) else "black"
                ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved: {out_path.name}")


def _load_per_class_iou_matrix(df, summary_dir):
    """
    Возвращает dict: (model, loss_config_name) → {class_idx: iou, ...}
    Читает из колонки best_val_per_class_iou (JSON строка) или из metrics JSON.
    """
    result = {}
    idx_to_class_global = {}

    for _, row in df.iterrows():
        model = row.get("model", "?")
        lc = row.get("loss_config_name", "?")

        # Пробуем колонку из summary CSV (JSON-строка)
        raw = row.get("best_val_per_class_iou")
        iou_list = None
        if isinstance(raw, str) and raw.strip() not in ("", "nan", "None"):
            try:
                iou_list = json.loads(raw)
            except Exception:
                pass

        idx_to_class = {}
        # Читаем из JSON-файла метрик
        mp = row.get("metrics_path")
        if mp and isinstance(mp, str):
            mp_obj = Path(mp)
            for cand in [mp_obj, summary_dir / "metrics" / mp_obj.name]:
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                    if iou_list is None:
                        iou_list = data.get("best_val_per_class_iou")
                    idx_to_class = data.get("idx_to_class", {})
                    idx_to_class_global.update(idx_to_class)
                    break
                except Exception:
                    pass

        if isinstance(iou_list, list):
            result[(model, lc)] = {i: float(v) for i, v in enumerate(iou_list) if v is not None}

    return result, idx_to_class_global


def _plot_per_class_heatmap_per_model(per_class_data, idx_to_class, loss_order, output_dir):
    """
    Для каждой модели рисует heatmap: ось X = loss_config, ось Y = class.
    """
    all_models = sorted({k[0] for k in per_class_data})
    all_classes = sorted({ci for d in per_class_data.values() for ci in d})

    for model in all_models:
        rows = {}
        for lc in loss_order:
            key = (model, lc)
            if key not in per_class_data:
                continue
            rows[lc] = {ci: per_class_data[key].get(ci, float("nan")) for ci in all_classes}
        if not rows:
            continue
        pivot = pd.DataFrame(rows, index=all_classes)
        pivot.index = [
            _get_class_label(ci, idx_to_class) for ci in pivot.index
        ]
        fname = f"matrix_per_class_iou__{_slug(model)}.png"
        _heatmap(
            pivot.T,          # строки=loss_configs, колонки=классы
            output_dir / fname,
            title=f"Per-class IoU — {model}",
            xlabel="Class",
            ylabel="Loss config",
            fmt=".2f",
            cmap="RdYlGn",
        )


def _plot_best_per_class_radar(per_class_data, idx_to_class, loss_order, output_dir):
    """
    Radar (spider) chart: для каждого loss_config — среднее по всем моделям.
    """
    all_classes = sorted({ci for d in per_class_data.values() for ci in d})
    if not all_classes:
        return

    class_labels = [_get_class_label(ci, idx_to_class) for ci in all_classes]
    N = len(all_classes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # замкнуть

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(loss_order)))

    for lc, color in zip(loss_order, colors):
        vals_per_model = []
        for model in {k[0] for k in per_class_data}:
            key = (model, lc)
            if key in per_class_data:
                vals_per_model.append([per_class_data[key].get(ci, float("nan")) for ci in all_classes])
        if not vals_per_model:
            continue
        mean_vals = np.nanmean(vals_per_model, axis=0).tolist()
        mean_vals += mean_vals[:1]
        ax.plot(angles, mean_vals, "o-", linewidth=1.8, label=_nice_loss_name(lc), color=color)
        ax.fill(angles, mean_vals, alpha=0.08, color=color)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class IoU radar (avg across models)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    out_path = output_dir / "matrix_radar_per_class_iou.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"  Saved: {out_path.name}")


def _slug(s):
    return str(s).lower().replace(" ", "_").replace("/", "_")


def plot_matrix(df, summary_path, output_dir):
    """Основная функция построения графиков для matrix-режима."""
    if "return_code" in df.columns:
        ok = df[df["return_code"] == 0].copy()
    else:
        ok = df.copy()
    if ok.empty:
        print("No successful runs.")
        return

    # Определяем порядок loss_configs (из данных + известный порядок)
    all_lc = ok["loss_config_name"].dropna().unique().tolist()
    loss_order = sorted(all_lc, key=_loss_order_key)

    # Порядок моделей
    model_order = ["pointnet", "pointnet++", "dgcnn",
                   "ldgcnn_gatv2", "ldgcnn_localwin", "ldgcnn_none"]
    all_models_data = ok["model"].dropna().unique().tolist()
    model_order = [m for m in model_order if m in all_models_data]
    model_order += [m for m in all_models_data if m not in model_order]

    print("Building matrix plots...")

    # Определяем режим: одна модель или несколько
    single_model = len(model_order) == 1
    model_label = model_order[0] if single_model else None

    def _title(base):
        """Добавляет название модели в заголовок при single-model режиме."""
        return f"{base} — {model_label}" if single_model else base

    # ── 1. Pivot-таблица mIoU ─────────────────────────────────────────────
    pivot_miou = None
    if "best_val_metric" in ok.columns:
        pivot_miou = (
            ok.pivot_table(index="model", columns="loss_config_name",
                           values="best_val_metric", aggfunc="max")
            .reindex(index=model_order, columns=loss_order)
        )

    # ── 2. mIoU: основной график ──────────────────────────────────────────
    if pivot_miou is not None and not pivot_miou.empty:
        ymin_miou = max(0, float(pivot_miou.min().min()) - 0.05)
        if single_model:
            # Один bar на loss config, без оси "Model"
            _simple_bar(
                pivot_miou.iloc[0].reindex(loss_order),
                output_dir / "matrix_bar_best_miou.png",
                title=_title("Best val mIoU по функциям потерь"),
                xlabel="Функция потерь", ylabel="mIoU",
                ylim=(ymin_miou, 1.0),
            )
        else:
            # Многомодельный режим: heatmap + два grouped-bar
            _heatmap(
                pivot_miou, output_dir / "matrix_heatmap_best_miou.png",
                _title("best val mIoU — model × loss config"),
                xlabel="Loss config", ylabel="Model",
            )
            _grouped_bar(
                pivot_miou, output_dir / "matrix_grouped_bar_miou_by_model.png",
                title=_title("Best val mIoU по моделям"),
                xlabel="Model", ylabel="mIoU",
                ylim=(ymin_miou, 1.0),
            )
            pivot_t = pivot_miou.T
            _grouped_bar(
                pivot_t, output_dir / "matrix_grouped_bar_miou_by_loss.png",
                title=_title("Best val mIoU по функциям потерь"),
                xlabel="Функция потерь", ylabel="mIoU",
                ylim=(max(0, float(pivot_t.min().min()) - 0.05), 1.0),
            )

    # ── 3. Accuracy ───────────────────────────────────────────────────────
    if "final_val_accuracy" in ok.columns:
        pivot_acc = (
            ok.pivot_table(index="model", columns="loss_config_name",
                           values="final_val_accuracy", aggfunc="max")
            .reindex(index=model_order, columns=loss_order)
        )
        if not pivot_acc.empty:
            if single_model:
                _simple_bar(
                    pivot_acc.iloc[0].reindex(loss_order),
                    output_dir / "matrix_bar_accuracy.png",
                    title=_title("Final val accuracy по функциям потерь"),
                    xlabel="Функция потерь", ylabel="Accuracy",
                    ylim=(max(0, float(pivot_acc.min().min()) - 0.05), 1.0),
                )
            else:
                _grouped_bar(
                    pivot_acc, output_dir / "matrix_grouped_bar_accuracy.png",
                    title="Final val accuracy по моделям",
                    xlabel="Model", ylabel="Accuracy",
                )
                _heatmap(
                    pivot_acc, output_dir / "matrix_heatmap_accuracy.png",
                    "Final val accuracy — model × loss config",
                    xlabel="Loss config", ylabel="Model",
                )

    # ── 4. Epoch time ─────────────────────────────────────────────────────
    if "mean_epoch_time_sec" in ok.columns:
        pivot_time = (
            ok.pivot_table(index="model", columns="loss_config_name",
                           values="mean_epoch_time_sec", aggfunc="mean")
            .reindex(index=model_order, columns=loss_order)
        )
        if not pivot_time.empty:
            if single_model:
                _simple_bar(
                    pivot_time.iloc[0].reindex(loss_order),
                    output_dir / "matrix_bar_epoch_time.png",
                    title=_title("Среднее время эпохи по функциям потерь"),
                    xlabel="Функция потерь", ylabel="Epoch time (s)",
                    fmt=".1f",
                )
            else:
                _grouped_bar(
                    pivot_time, output_dir / "matrix_grouped_bar_epoch_time.png",
                    title="Mean epoch time (sec) по моделям",
                    xlabel="Model", ylabel="Epoch time (s)",
                    fmt=".1f", annot=True,
                )

    # ── 5. VRAM ───────────────────────────────────────────────────────────
    if "max_peak_vram_mb" in ok.columns:
        pivot_vram = (
            ok.pivot_table(index="model", columns="loss_config_name",
                           values="max_peak_vram_mb", aggfunc="max")
            .reindex(index=model_order, columns=loss_order)
        )
        if not pivot_vram.empty:
            if single_model:
                _simple_bar(
                    pivot_vram.iloc[0].reindex(loss_order),
                    output_dir / "matrix_bar_vram.png",
                    title=_title("Peak VRAM по функциям потерь"),
                    xlabel="Функция потерь", ylabel="Peak VRAM (MB)",
                    fmt=".0f",
                )
            else:
                _heatmap(
                    pivot_vram, output_dir / "matrix_heatmap_vram.png",
                    "Peak VRAM (MB) — model × loss config",
                    xlabel="Loss config", ylabel="Model",
                    fmt=".0f", cmap="Blues",
                )

    # ── 6. Best epoch ─────────────────────────────────────────────────────
    if "best_epoch" in ok.columns:
        pivot_ep = (
            ok.pivot_table(index="model", columns="loss_config_name",
                           values="best_epoch", aggfunc="mean")
            .reindex(index=model_order, columns=loss_order)
        )
        if not pivot_ep.empty:
            if single_model:
                _simple_bar(
                    pivot_ep.iloc[0].reindex(loss_order),
                    output_dir / "matrix_bar_best_epoch.png",
                    title=_title("Эпоха лучшего mIoU по функциям потерь"),
                    xlabel="Функция потерь", ylabel="Epoch",
                    fmt=".0f",
                )
            else:
                _grouped_bar(
                    pivot_ep, output_dir / "matrix_grouped_bar_best_epoch.png",
                    title="Best epoch (когда достигнут лучший mIoU)",
                    xlabel="Model", ylabel="Epoch",
                    fmt=".0f", annot=True,
                )

    # ── 7. Per-class IoU heatmap и radar ──────────────────────────────────
    per_class_data, idx_to_class = _load_per_class_iou_matrix(ok, summary_path.parent)
    if per_class_data:
        _plot_per_class_heatmap_per_model(per_class_data, idx_to_class, loss_order, output_dir)
        if not single_model:
            # Radar усредняет по моделям — при одной модели избыточен
            _plot_best_per_class_radar(per_class_data, idx_to_class, loss_order, output_dir)

    print(f"Matrix plots saved to: {output_dir}")


# ─────────────────────────── entry point ──────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Plot sweep summary metrics")
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--x_col", type=str, default=None,
        choices=["batch_size", "class_balance_beta"],
        help="Явно задать ось для single-axis режима.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).resolve()
    df = pd.read_csv(summary_path)

    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else summary_path.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = _detect_mode(df)
    print(f"Plot mode: {mode}")

    if mode == "matrix":
        plot_matrix(df, summary_path, output_dir)
    else:
        plot_single_axis(df, summary_path, output_dir)


if __name__ == "__main__":
    main()
