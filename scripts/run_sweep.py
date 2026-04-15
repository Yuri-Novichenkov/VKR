"""
Batch-size sweep runner based on YAML config.

Runs multiple train.py launches, stores logs/metrics in unique folders,
and writes a consolidated runs_summary.csv for plotting/comparison.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _slug(value):
    text = str(value).strip().lower()
    for ch in (" ", "/", "\\", ":", ","):
        text = text.replace(ch, "_")
    return text


def _format_cli_value(v):
    if isinstance(v, bool):
        return None
    return str(v)


def _build_experiment_dir(base_dir: Path, cfg: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = _slug(cfg.get("model", "model"))
    dataset = _slug(cfg.get("dataset", "dataset"))
    exp_name = _slug(cfg.get("experiment_name", "batch_sweep"))
    run_group = _slug(cfg.get("run_group", "default"))
    folder = f"{stamp}_{exp_name}_{model}_{dataset}_{run_group}"
    out_dir = base_dir / folder
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _build_train_command(
    config: dict,
    python_executable: str,
    batch_size: int,
    run_name: str,
    metrics_json_path: Path,
) -> list[str]:
    common_args = deepcopy(config.get("common_args", {}))
    train_args = deepcopy(config.get("train_args", {}))
    script = config.get("train_script", "scripts/train.py")

    cmd = [python_executable, script]
    fixed = {
        "model": config["model"],
        "task": config.get("task", "segmentation"),
        "dataset": config["dataset"],
        "batch_size": batch_size,
        "run_name": run_name,
        "metrics_json_path": str(metrics_json_path),
    }
    merged = {}
    merged.update(common_args)
    merged.update(train_args)
    merged.update(fixed)
    if "experiment_name" in config:
        merged["experiment_name"] = config["experiment_name"]

    for key, value in merged.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        cmd.append(flag)
        cmd.append(_format_cli_value(value))
    return cmd


def _read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _run_one(cmd: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(cmd) + "\n\n")
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def _write_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Run batch-size sweep from YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_root", type=str, default="experiments/batch_size", help="Root folder for sweep outputs")
    parser.add_argument(
        "--python_executable",
        type=str,
        default=None,
        help="Python executable for child train.py runs (default: from config or current interpreter)",
    )
    parser.add_argument("--no_auto_plot", action="store_true", help="Disable automatic plotting after sweep")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not run")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    batch_sizes = cfg.get("batch_sizes", [])
    if not batch_sizes:
        raise ValueError("config.batch_sizes must contain at least one value")

    python_executable = args.python_executable or cfg.get("python_executable") or sys.executable

    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = _slug(cfg.get("model", "model"))
        dataset = _slug(cfg.get("dataset", "dataset"))
        exp_name = _slug(cfg.get("experiment_name", "batch_sweep"))
        run_group = _slug(cfg.get("run_group", "default"))
        folder = f"{stamp}_{exp_name}_{model}_{dataset}_{run_group}"
        exp_dir = output_root / folder
        logs_dir = exp_dir / "logs"
        metrics_dir = exp_dir / "metrics"
        plots_dir = exp_dir / "plots"
    else:
        exp_dir = _build_experiment_dir(output_root, cfg)
        logs_dir = exp_dir / "logs"
        metrics_dir = exp_dir / "metrics"
        plots_dir = exp_dir / "plots"
        logs_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

    # Persist full resolved config for reproducibility.
    resolved_cfg = deepcopy(cfg)
    resolved_cfg["resolved_python_executable"] = python_executable
    resolved_cfg["auto_plot_enabled"] = not args.no_auto_plot
    resolved_cfg_path = exp_dir / "config_resolved.yaml"
    if not args.dry_run:
        with resolved_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_cfg, f, sort_keys=False, allow_unicode=True)

    summary_rows = []
    print(f"Sweep output directory: {exp_dir}")
    for bs in batch_sizes:
        bs = int(bs)
        run_tag = cfg.get("run_group", "bs_sweep")
        run_name = f"{run_tag}__bs{bs}__seed{cfg.get('common_args', {}).get('seed', 42)}"
        log_path = logs_dir / f"train_bs{bs}.log"
        metrics_path = metrics_dir / f"metrics_bs{bs}.json"
        cmd = _build_train_command(cfg, python_executable, bs, run_name, metrics_path)

        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
            continue

        print(f"[RUN] batch_size={bs}")
        t0 = time.perf_counter()
        return_code = _run_one(cmd, log_path)
        wall_time_sec = float(time.perf_counter() - t0)

        row = {
            "batch_size": bs,
            "return_code": return_code,
            "wall_time_sec": wall_time_sec,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "run_name": run_name,
        }
        if return_code == 0 and metrics_path.exists():
            data = _read_metrics(metrics_path)
            row.update(
                {
                    "best_val_metric": data.get("best_val_metric"),
                    "best_epoch": data.get("best_epoch"),
                    "total_train_time_sec": data.get("total_train_time_sec"),
                    "mean_epoch_time_sec": data.get("mean_epoch_time_sec"),
                    "max_peak_vram_mb": data.get("max_peak_vram_mb"),
                    "final_val_accuracy": data.get("final_val_accuracy"),
                    "final_val_miou": data.get("final_val_miou"),
                    "train_drop_last": data.get("train_drop_last"),
                    "train_dataset_size": data.get("train_dataset_size"),
                    "train_items_per_epoch": data.get("train_items_per_epoch"),
                    "train_steps_per_epoch": data.get("train_steps_per_epoch"),
                    "effective_drop_ratio": data.get("effective_drop_ratio"),
                    "num_points": data.get("num_points"),
                }
            )
        else:
            print(f"  FAILED (code={return_code}) -> {log_path}")

        summary_rows.append(row)

    summary_csv = exp_dir / "runs_summary.csv"
    if not args.dry_run:
        _write_summary(summary_rows, summary_csv)
        print(f"Saved summary: {summary_csv}")
        print(f"Logs directory: {logs_dir}")
        print(f"Metrics directory: {metrics_dir}")
        print(f"Plots directory: {plots_dir}")

        if summary_rows and not args.no_auto_plot:
            plot_script = ROOT / "scripts" / "plot_sweep_results.py"
            plot_cmd = [
                python_executable,
                str(plot_script),
                "--summary_csv",
                str(summary_csv),
                "--output_dir",
                str(plots_dir),
            ]
            print("[PLOT]", " ".join(plot_cmd))
            plot_rc = subprocess.call(plot_cmd, cwd=str(ROOT))
            if plot_rc != 0:
                print(f"WARN: auto-plot failed with exit code {plot_rc}.")


if __name__ == "__main__":
    main()
