"""
Benchmark инференса: сравнение скорости и потребления памяти
для разных режимов оптимизации.

Режимы:
  fp32          — базовый FP32 (без оптимизаций)
  fp16          — FP16 (model.half())
  compile_fp32  — torch.compile + FP32
  compile_fp16  — torch.compile + FP16
  onnx_cpu      — ONNX Runtime, CPU provider
  onnx_gpu      — ONNX Runtime, CUDA provider

Пример запуска:
  # Все режимы для одного чекпоинта
  python scripts/benchmark_inference.py \
      --checkpoint checkpoints/loss_sweep/pointnet/segmentation/cb_effective_b0p99999/mar16/best_model.pth

  # Только конкретные режимы
  python scripts/benchmark_inference.py \
      --checkpoint checkpoints/loss_sweep/dgcnn/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth \
      --modes fp32 fp16 compile_fp16

  # Несколько чекпоинтов сразу (батч-режим)
  python scripts/benchmark_inference.py \
      --checkpoint_dir checkpoints/loss_sweep \
      --output_csv results/benchmark_mar16.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model

# ──────────────────────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────────────────────

ALL_MODES = ["fp32", "fp16", "compile_fp32", "compile_fp16", "onnx_cpu", "onnx_gpu"]
DEFAULT_MODES = ["fp32", "fp16", "compile_fp32", "compile_fp16"]

WARMUP_ITERS = 10   # прогрев JIT/compile кэша
BENCH_ITERS  = 50   # измерительные итерации


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────────────────────

def _load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def _build_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module:
    model = build_model(
        model_type=ckpt.get("model_type", "pointnet"),
        task=ckpt.get("task", "segmentation"),
        num_classes=ckpt["num_classes"],
        num_features=ckpt["num_features"],
        k=ckpt.get("k", 20),
        k_small=ckpt.get("k_small", 20),
        k_large=ckpt.get("k_large", 40),
        attention_type=ckpt.get("attention_type", "none"),
        attention_k=ckpt.get("attention_k", 16),
        attention_heads=ckpt.get("attention_heads", 4),
        attention_dropout=ckpt.get("attention_dropout", 0.1),
        pt_k=ckpt.get("pt_k", 16),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _make_input(batch_size: int, num_points: int, num_features: int,
                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Синтетический батч (B, N, C)."""
    return torch.randn(batch_size, num_points, num_features,
                       device=device, dtype=dtype)


def _reset_vram(device: torch.device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def _peak_vram_mb(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device) / 1024 ** 2
    return 0.0


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ──────────────────────────────────────────────────────────────────────────────
# Замер одного режима
# ──────────────────────────────────────────────────────────────────────────────

def _bench_pytorch(model: torch.nn.Module, x: torch.Tensor,
                   device: torch.device) -> dict:
    """Запускает warmup + bench и возвращает статистику."""
    with torch.no_grad():
        # Прогрев
        for _ in range(WARMUP_ITERS):
            _ = model(x)
            _sync(device)

        _reset_vram(device)
        times = []
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            _ = model(x)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    peak_mb = _peak_vram_mb(device)
    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "peak_vram_mb": peak_mb,
    }


def _bench_onnx(onnx_path: str, x_np: np.ndarray, provider: str) -> dict:
    """Запускает ONNX Runtime и возвращает статистику."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=[provider],
        )
    except Exception as e:
        return {"error": str(e)}

    input_name = sess.get_inputs()[0].name

    # Прогрев
    for _ in range(WARMUP_ITERS):
        sess.run(None, {input_name: x_np})

    times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x_np})
        times.append((time.perf_counter() - t0) * 1000)

    times = np.array(times)
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "peak_vram_mb": 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# ONNX экспорт
# ──────────────────────────────────────────────────────────────────────────────

def _export_onnx(model: torch.nn.Module, x: torch.Tensor,
                 onnx_path: str) -> bool:
    """Экспортирует модель в ONNX. Возвращает True при успехе."""
    try:
        torch.onnx.export(
            model,
            x,
            onnx_path,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
        return True
    except Exception as e:
        print(f"    [ONNX export error] {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Основная функция бенчмарка одного чекпоинта
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_checkpoint(ckpt_path: str, modes: list[str],
                          batch_size: int, num_points: int,
                          device: torch.device) -> list[dict]:
    """
    Запускает все запрошенные режимы для одного чекпоинта.
    Возвращает список строк (по одной на режим) для CSV.
    """
    print(f"\n{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = _load_checkpoint(ckpt_path, device)
    model_type   = ckpt.get("model_type", "pointnet")
    num_features = ckpt["num_features"]
    dataset      = ckpt.get("dataset", "?")
    num_params   = sum(p.numel() for p in
                       _build_from_ckpt(ckpt, torch.device("cpu")).parameters())

    print(f"Model: {model_type}  |  features: {num_features}  "
          f"|  params: {num_params:,}  |  dataset: {dataset}")

    rows = []
    onnx_path = None  # кэшируем путь к экспортированному ONNX

    for mode in modes:
        print(f"  [{mode}] ", end="", flush=True)

        row = {
            "model": model_type,
            "checkpoint": ckpt_path,
            "dataset": dataset,
            "mode": mode,
            "batch_size": batch_size,
            "num_points": num_points,
            "num_params": num_params,
        }

        try:
            if mode in ("fp32", "fp16", "compile_fp32", "compile_fp16"):
                model = _build_from_ckpt(ckpt, device)

                if "fp16" in mode:
                    model = model.half()
                    dtype = torch.float16
                else:
                    dtype = torch.float32

                if "compile" in mode:
                    # backend='eager' работает везде (нет зависимости от triton/Windows)
                    # backend='inductor' быстрее, но требует Linux + triton
                    compile_backend = "inductor" if sys.platform != "win32" else "eager"
                    model = torch.compile(model, backend=compile_backend)
                    print(f"(compile backend={compile_backend}) ", end="", flush=True)

                x = _make_input(batch_size, num_points, num_features, device, dtype)
                stats = _bench_pytorch(model, x, device)

            elif mode in ("onnx_cpu", "onnx_gpu"):
                # Один раз экспортируем ONNX (fp32)
                if onnx_path is None:
                    _onnx_model = _build_from_ckpt(ckpt, torch.device("cpu"))
                    x_cpu = _make_input(batch_size, num_points, num_features,
                                        torch.device("cpu"), torch.float32)
                    onnx_path = str(ROOT / "tmp_bench_model.onnx")
                    ok = _export_onnx(_onnx_model, x_cpu, onnx_path)
                    if not ok:
                        onnx_path = None

                if onnx_path is None:
                    stats = {"error": "ONNX export failed"}
                else:
                    x_np = _make_input(
                        batch_size, num_points, num_features,
                        torch.device("cpu"), torch.float32
                    ).numpy()
                    provider = ("CUDAExecutionProvider" if mode == "onnx_gpu"
                                else "CPUExecutionProvider")
                    stats = _bench_onnx(onnx_path, x_np, provider)
            else:
                stats = {"error": f"unknown mode {mode}"}

        except Exception as e:
            stats = {"error": str(e)}

        row.update(stats)

        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            throughput = batch_size / (stats["mean_ms"] / 1000)
            row["throughput_clouds_per_sec"] = round(throughput, 2)
            print(f"latency={stats['mean_ms']:.1f}±{stats['std_ms']:.1f} ms  "
                  f"throughput={throughput:.1f} clouds/s  "
                  f"VRAM={stats['peak_vram_mb']:.0f} MB")

        rows.append(row)

    # Удаляем временный ONNX файл
    if onnx_path and Path(onnx_path).exists():
        Path(onnx_path).unlink()

    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Авто-поиск лучших чекпоинтов
# ──────────────────────────────────────────────────────────────────────────────

# Лучшие конфигурации каждой модели на Mar16 из loss_sweep
BEST_CHECKPOINTS_MAR16 = {
    "pointnet":       "checkpoints/loss_sweep/pointnet/segmentation/cb_effective_b0p99999/mar16/best_model.pth",
    "pointnet++":     "checkpoints/loss_sweep/pointnet++/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth",
    "dgcnn":          "checkpoints/loss_sweep/dgcnn/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth",
    "ldgcnn_gatv2":   "checkpoints/loss_sweep/ldgcnn/segmentation/attn_gatv2_k16_h4_d0p1__cb_effective_b0p99999/mar16/best_model.pth",
    "pointtransformer":"checkpoints/loss_sweep/pointtransformer/segmentation/cb_effective_b0p99999/mar16/best_model.pth",
    "ldgcnn_flash":   "checkpoints/loss_sweep/ldgcnn_flash/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth",
}


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark inference для моделей сегментации облаков точек")

    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--checkpoint", type=str, default=None,
                           help="Путь к одному чекпоинту")
    src_group.add_argument("--all_mar16", action="store_true",
                           help="Запустить все 6 лучших моделей Mar16 из loss_sweep")

    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=ALL_MODES, metavar="MODE",
                        help=f"Режимы оптимизации (default: {DEFAULT_MODES}). "
                             f"Все режимы: {ALL_MODES}")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Размер батча для бенчмарка (default: 1 — реальный инференс)")
    parser.add_argument("--num_points", type=int, default=4096,
                        help="Точек в облаке (default: 4096)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Устройство (default: auto → cuda если есть)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Путь для сохранения результатов в CSV")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Путь для сохранения результатов в JSON")

    args = parser.parse_args()

    # Устройство
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Список чекпоинтов
    if args.all_mar16:
        checkpoints = [str(ROOT / p) for p in BEST_CHECKPOINTS_MAR16.values()
                       if (ROOT / p).exists()]
        missing = [name for name, p in BEST_CHECKPOINTS_MAR16.items()
                   if not (ROOT / p).exists()]
        if missing:
            print(f"Предупреждение: не найдены чекпоинты для: {missing}")
    elif args.checkpoint:
        checkpoints = [args.checkpoint]
    else:
        parser.error("Укажите --checkpoint <path> или --all_mar16")
        return

    # Запуск
    all_rows = []
    for ckpt_path in checkpoints:
        rows = benchmark_checkpoint(
            ckpt_path,
            modes=args.modes,
            batch_size=args.batch_size,
            num_points=args.num_points,
            device=device,
        )
        all_rows.extend(rows)

    # Итоговая таблица
    print(f"\n{'='*60}")
    print("ИТОГО:")
    print(f"{'='*60}")
    _print_table(all_rows)

    # Сохранение
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        _save_csv(all_rows, args.output_csv)
        print(f"\nCSV сохранён: {args.output_csv}")

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"JSON сохранён: {args.output_json}")


def _print_table(rows: list[dict]):
    header = f"{'Model':<20} {'Mode':<15} {'Latency ms':>12} {'±':>6} {'Throughput':>12} {'VRAM MB':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r.get('model','?'):<20} {r['mode']:<15} {'ERROR':>12}  {r['error']}")
        else:
            print(f"{r.get('model','?'):<20} {r['mode']:<15} "
                  f"{r.get('mean_ms',0):>12.1f} "
                  f"{r.get('std_ms',0):>6.1f} "
                  f"{r.get('throughput_clouds_per_sec',0):>12.1f} "
                  f"{r.get('peak_vram_mb',0):>9.0f}")


def _save_csv(rows: list[dict], path: str):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
