"""
Build comparative plots for sweep runs from runs_summary.csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_line(df, x_col, y_col, out_path, title, ylabel):
    if y_col not in df.columns:
        return
    part = df[[x_col, y_col]].dropna().sort_values(x_col)
    if part.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(part[x_col], part[y_col], marker="o")
    for _, row in part.iterrows():
        plt.annotate(f"{row[y_col]:.3f}" if isinstance(row[y_col], float) else str(row[y_col]), (row[x_col], row[y_col]))
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _safe_divide(numerator_series, denominator_series):
    out = numerator_series / denominator_series
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot sweep summary metrics")
    parser.add_argument("--summary_csv", type=str, required=True, help="Path to runs_summary.csv")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots (default: <summary_dir>/plots)")
    args = parser.parse_args()

    summary_path = Path(args.summary_csv).resolve()
    df = pd.read_csv(summary_path)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (summary_path.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep only successful runs for metric plots.
    if "return_code" in df.columns:
        ok_df = df[df["return_code"] == 0].copy()
    else:
        ok_df = df.copy()

    if ok_df.empty:
        print("No successful runs found in summary.")
        return

    x_col = "batch_size"
    _plot_line(ok_df, x_col, "best_val_metric", output_dir / "batch_vs_best_val_metric.png", "Batch size vs best validation metric", "best_val_metric")
    _plot_line(ok_df, x_col, "final_val_miou", output_dir / "batch_vs_final_val_miou.png", "Batch size vs final val mIoU", "final_val_miou")
    _plot_line(ok_df, x_col, "final_val_accuracy", output_dir / "batch_vs_final_val_accuracy.png", "Batch size vs final val accuracy", "final_val_accuracy")
    _plot_line(ok_df, x_col, "mean_epoch_time_sec", output_dir / "batch_vs_mean_epoch_time_sec.png", "Batch size vs mean epoch time", "mean_epoch_time_sec")
    _plot_line(ok_df, x_col, "total_train_time_sec", output_dir / "batch_vs_total_train_time_sec.png", "Batch size vs total train time", "total_train_time_sec")
    _plot_line(ok_df, x_col, "max_peak_vram_mb", output_dir / "batch_vs_max_peak_vram_mb.png", "Batch size vs peak VRAM", "max_peak_vram_mb")

    # Throughput: честно через обработанные объекты за эпоху.
    if {"mean_epoch_time_sec", "train_items_per_epoch"}.issubset(ok_df.columns):
        part = ok_df[[x_col, "mean_epoch_time_sec", "train_items_per_epoch"]].dropna().copy()
        part["samples_per_sec"] = _safe_divide(part["train_items_per_epoch"], part["mean_epoch_time_sec"])
        _plot_line(
            part,
            x_col,
            "samples_per_sec",
            output_dir / "batch_vs_samples_per_sec.png",
            "Batch size vs samples per second",
            "samples_per_sec",
        )

    # Throughput по точкам/сек (если есть num_points).
    if {"mean_epoch_time_sec", "train_items_per_epoch", "num_points"}.issubset(ok_df.columns):
        part = ok_df[[x_col, "mean_epoch_time_sec", "train_items_per_epoch", "num_points"]].dropna().copy()
        part["points_per_sec"] = _safe_divide(part["train_items_per_epoch"] * part["num_points"], part["mean_epoch_time_sec"])
        _plot_line(
            part,
            x_col,
            "points_per_sec",
            output_dir / "batch_vs_points_per_sec.png",
            "Batch size vs points per second",
            "points_per_sec",
        )

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
