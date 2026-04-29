"""
Sweep runner based on YAML config.

Поддерживает три режима (определяются автоматически по ключам YAML):

1. **batch_sizes** — вариация --batch_size для одной модели (исторический).
2. **class_balance_betas** — вариация --class_balance_beta для одной модели.
3. **matrix** — полная матрица ``models`` × ``loss_configs``:
   - ``models``: список словарей {model, task?, train_args?}
   - ``loss_configs``: список словарей {name, loss_type, focal_gamma?,
     class_balance?, class_balance_beta?}

В режиме matrix каждая комбинация (модель × loss_config) запускается как
отдельный train.py. Результаты пишутся в runs_summary.csv с колонками
``model``, ``loss_config_name``, ``loss_type``, что позволяет plot_sweep_results
строить grouped-bar и heatmap графики.
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


# ─────────────────────────── режим свипа ──────────────────────────────────


def _determine_sweep_mode(cfg: dict) -> str:
    """Определяет режим: 'matrix', 'batch_size' или 'class_balance_beta'."""
    has_models = bool(cfg.get("models"))
    has_loss_configs = bool(cfg.get("loss_configs"))
    has_batch_sizes = bool(cfg.get("batch_sizes"))
    has_betas = bool(cfg.get("class_balance_betas"))

    if has_models and has_loss_configs:
        if has_batch_sizes or has_betas:
            raise ValueError(
                "В matrix-режиме (models + loss_configs) нельзя одновременно "
                "указывать batch_sizes или class_balance_betas."
            )
        return "matrix"
    if has_batch_sizes and has_betas:
        raise ValueError(
            "config может задавать только одну ось свипа: "
            "batch_sizes или class_balance_betas, но не обе сразу."
        )
    if has_batch_sizes:
        return "batch_size"
    if has_betas:
        return "class_balance_beta"
    raise ValueError(
        "config должен задавать либо (models + loss_configs), "
        "либо batch_sizes, либо class_balance_betas."
    )


# ─────────────────────────── single-axis helpers ──────────────────────────


def _slug_value_single(axis: str, value) -> str:
    if axis == "batch_size":
        return f"bs{int(value)}"
    if axis == "class_balance_beta":
        return "b" + str(value).replace(".", "p").replace("-", "neg")
    return _slug(value)


def _build_experiment_dir(base_dir: Path, cfg: dict) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = _slug(cfg.get("model", cfg.get("models", [{}])[0].get("model", "model")))
    dataset = _slug(cfg.get("dataset", "dataset"))
    exp_name = _slug(cfg.get("experiment_name", "sweep"))
    run_group = _slug(cfg.get("run_group", "default"))
    folder = f"{stamp}_{exp_name}_{model}_{dataset}_{run_group}"
    out_dir = base_dir / folder
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def _build_train_command(
    config: dict,
    python_executable: str,
    extra_args: dict,
    run_name: str,
    metrics_json_path: Path,
    model_override: dict | None = None,
) -> list[str]:
    """
    Собирает CLI-команду. extra_args перекрывает common_args/train_args.
    model_override — словарь {model, task?, train_args?} для matrix-режима.
    """
    common_args = deepcopy(config.get("common_args", {}))
    train_args = deepcopy(config.get("train_args", {}))
    script = config.get("train_script", "scripts/train.py")

    # В matrix-режиме модель задаётся через model_override, а не через cfg["model"]
    if model_override:
        fixed_model = model_override.get("model", "pointnet")
        fixed_task = model_override.get("task", config.get("task", "segmentation"))
        # Индивидуальные train_args модели перекрывают общие
        model_train_args = deepcopy(model_override.get("train_args", {}))
        train_args.update(model_train_args)
    else:
        fixed_model = config["model"]
        fixed_task = config.get("task", "segmentation")

    cmd = [python_executable, script]
    fixed = {
        "model": fixed_model,
        "task": fixed_task,
        "dataset": config["dataset"],
        "run_name": run_name,
        "metrics_json_path": str(metrics_json_path),
    }
    merged = {}
    merged.update(common_args)
    merged.update(train_args)
    merged.update(fixed)
    merged.update(extra_args)
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


# ─────────────────────────── runner ───────────────────────────────────────


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


def _read_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_summary(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    headers = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ─────────────────────────── matrix sweep ─────────────────────────────────


def _run_matrix_sweep(cfg, exp_dir, logs_dir, metrics_dir, python_executable, dry_run):
    """Запускает полную матрицу models × loss_configs."""
    models_list = cfg["models"]          # [{model, task?, train_args?}, ...]
    loss_configs = cfg["loss_configs"]   # [{name, loss_type, ...}, ...]
    summary_rows = []
    seed = cfg.get("common_args", {}).get("seed", 42)

    total = len(models_list) * len(loss_configs)
    run_idx = 0

    for model_cfg in models_list:
        model_name = model_cfg["model"]
        # Если задано поле name — используем его для уникальной идентификации варианта
        model_display_name = model_cfg.get("name", model_name)
        for lc in loss_configs:
            run_idx += 1
            lc_name = lc["name"]

            # Собираем extra_args из loss_config (исключаем служебное поле name)
            loss_extra = {k: v for k, v in lc.items() if k != "name"}

            # Slug для имён файлов — используем display_name чтобы различать варианты
            slug = f"{_slug(model_display_name)}__{_slug(lc_name)}"
            run_name = (
                f"{cfg.get('run_group', 'loss_sweep')}"
                f"__{slug}__seed{seed}"
            )
            log_path = logs_dir / f"train_{slug}.log"
            metrics_path = metrics_dir / f"metrics_{slug}.json"

            cmd = _build_train_command(
                cfg,
                python_executable,
                extra_args=loss_extra,
                run_name=run_name,
                metrics_json_path=metrics_path,
                model_override=model_cfg,
            )

            print(f"[{run_idx}/{total}] model={model_name}, loss={lc_name}")
            if dry_run:
                print("  [DRY RUN]", " ".join(cmd))
                continue

            t0 = time.perf_counter()
            return_code = _run_one(cmd, log_path)
            wall_time_sec = float(time.perf_counter() - t0)

            row = {
                "model": model_display_name,
                "loss_config_name": lc_name,
                "loss_type": lc.get("loss_type", "ce"),
                "class_balance": lc.get("class_balance", "none"),
                "class_balance_beta": lc.get("class_balance_beta"),
                "return_code": return_code,
                "wall_time_sec": wall_time_sec,
                "log_path": str(log_path),
                "metrics_path": str(metrics_path),
                "run_name": run_name,
            }
            if return_code == 0 and metrics_path.exists():
                data = _read_metrics(metrics_path)
                row.update({
                    "best_val_metric": data.get("best_val_metric"),
                    "best_epoch": data.get("best_epoch"),
                    "total_train_time_sec": data.get("total_train_time_sec"),
                    "mean_epoch_time_sec": data.get("mean_epoch_time_sec"),
                    "max_peak_vram_mb": data.get("max_peak_vram_mb"),
                    "final_val_accuracy": data.get("final_val_accuracy"),
                    "final_val_miou": data.get("final_val_miou"),
                    "best_val_miou": data.get("best_val_miou"),
                    "num_points": data.get("num_points"),
                    "batch_size": data.get("batch_size"),
                    "best_val_per_class_iou": json.dumps(data.get("best_val_per_class_iou")),
                })
            else:
                print(f"  FAILED (code={return_code}) -> {log_path}")
            summary_rows.append(row)

    return summary_rows


# ─────────────────────────── single-axis sweep ────────────────────────────


_DEFAULT_OUTPUT_ROOT_PER_AXIS = {
    "batch_size": "experiments/batch_size",
    "class_balance_beta": "experiments/class_balance_beta",
}


def _run_single_axis_sweep(cfg, sweep_axis, sweep_values, exp_dir, logs_dir, metrics_dir, python_executable, dry_run):
    summary_rows = []
    seed = cfg.get("common_args", {}).get("seed", 42)
    total = len(sweep_values)

    for i, value in enumerate(sweep_values, 1):
        slug = _slug_value_single(sweep_axis, value)
        run_tag = cfg.get("run_group", f"{sweep_axis}_sweep")
        run_name = f"{run_tag}__{slug}__seed{seed}"
        log_path = logs_dir / f"train_{slug}.log"
        metrics_path = metrics_dir / f"metrics_{slug}.json"
        axis_overrides = {sweep_axis: value}
        cmd = _build_train_command(cfg, python_executable, axis_overrides, run_name, metrics_path)

        print(f"[{i}/{total}] {sweep_axis}={value}")
        if dry_run:
            print("  [DRY RUN]", " ".join(cmd))
            continue

        t0 = time.perf_counter()
        return_code = _run_one(cmd, log_path)
        wall_time_sec = float(time.perf_counter() - t0)

        row = {
            sweep_axis: value,
            "return_code": return_code,
            "wall_time_sec": wall_time_sec,
            "log_path": str(log_path),
            "metrics_path": str(metrics_path),
            "run_name": run_name,
        }
        if return_code == 0 and metrics_path.exists():
            data = _read_metrics(metrics_path)
            row.update({
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
                "batch_size": data.get("batch_size"),
                "class_balance": data.get("class_balance"),
                "class_balance_beta": data.get("class_balance_beta"),
            })
        else:
            print(f"  FAILED (code={return_code}) -> {log_path}")
        summary_rows.append(row)

    return summary_rows


# ─────────────────────────── main ─────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run sweep from YAML config (batch_sizes, class_balance_betas, or models × loss_configs)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root folder for sweep outputs (default: experiments/<mode>).",
    )
    parser.add_argument(
        "--python_executable",
        type=str,
        default=None,
        help="Python executable for child train.py runs (default: current interpreter)",
    )
    parser.add_argument("--no_auto_plot", action="store_true", help="Disable automatic plotting after sweep")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not run")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sweep_mode = _determine_sweep_mode(cfg)
    print(f"Sweep mode: {sweep_mode}")

    python_executable = args.python_executable or cfg.get("python_executable") or sys.executable

    # Определяем output_root
    if args.output_root:
        output_root = (ROOT / args.output_root).resolve()
    elif sweep_mode == "matrix":
        output_root = ROOT / "experiments" / "loss_sweep"
    else:
        output_root = ROOT / _DEFAULT_OUTPUT_ROOT_PER_AXIS.get(sweep_mode, "experiments/sweep")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        exp_dir = output_root / "dry_run"
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

    resolved_cfg = deepcopy(cfg)
    resolved_cfg["resolved_python_executable"] = python_executable
    resolved_cfg["auto_plot_enabled"] = not args.no_auto_plot
    resolved_cfg["sweep_mode"] = sweep_mode
    if not args.dry_run:
        with (exp_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(resolved_cfg, f, sort_keys=False, allow_unicode=True)

    print(f"Sweep output directory: {exp_dir}")

    if sweep_mode == "matrix":
        summary_rows = _run_matrix_sweep(cfg, exp_dir, logs_dir, metrics_dir, python_executable, args.dry_run)
    else:
        # single-axis mode
        bs_vals = cfg.get("batch_sizes")
        beta_vals = cfg.get("class_balance_betas")
        if bs_vals:
            sweep_values = [int(v) for v in bs_vals]
        else:
            sweep_values = [float(v) for v in beta_vals]
        print(f"Sweep axis: {sweep_mode}, values: {sweep_values}")
        summary_rows = _run_single_axis_sweep(
            cfg, sweep_mode, sweep_values, exp_dir, logs_dir, metrics_dir, python_executable, args.dry_run
        )

    if args.dry_run:
        return

    summary_csv = exp_dir / "runs_summary.csv"
    _write_summary(summary_rows, summary_csv)
    print(f"Saved summary: {summary_csv}")
    print(f"Logs:    {logs_dir}")
    print(f"Metrics: {metrics_dir}")

    if summary_rows and not args.no_auto_plot:
        plot_script = ROOT / "scripts" / "plot_sweep_results.py"
        plot_cmd = [
            python_executable, str(plot_script),
            "--summary_csv", str(summary_csv),
            "--output_dir", str(plots_dir),
        ]
        print("[PLOT]", " ".join(plot_cmd))
        plot_rc = subprocess.call(plot_cmd, cwd=str(ROOT))
        if plot_rc != 0:
            print(f"WARN: auto-plot failed with exit code {plot_rc}.")


if __name__ == "__main__":
    main()
