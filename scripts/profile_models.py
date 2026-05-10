"""
Профилирование всех моделей:
  - Расширенная таблица параметров (params, FLOPs, VRAM fp32/fp16, размер .pth)
  - Текстовая таблица топ-операций (PyTorch Profiler)
  - Chrome trace JSON (открывать в chrome://tracing)

Использование:
    python scripts/profile_models.py --checkpoints_dir checkpoints --output_dir results/profiler
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model

# Соответствие model_type → display name
MODEL_DISPLAY = {
    "pointnet":           "PointNet",
    "pointnet2":          "PointNet++",
    "dgcnn":              "DGCNN",
    "ldgcnn":             "LDGCNN",
    "point_transformer":  "PointTransformer",
    "ldgcnn_flash":       "LDGCNNFlash",
}

ORDER = ["pointnet", "pointnet2", "dgcnn", "ldgcnn", "point_transformer", "ldgcnn_flash"]


def load_checkpoint_meta(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "model_type":      ckpt.get("model_type", "pointnet"),
        "num_classes":     ckpt["num_classes"],
        "num_features":    ckpt["num_features"],
        "k":               ckpt.get("k", 20),
        "k_small":         ckpt.get("k_small", 20),
        "k_large":         ckpt.get("k_large", 40),
        "attention_type":  ckpt.get("attention_type", "none"),
        "attention_k":     ckpt.get("attention_k", 16),
        "attention_heads": ckpt.get("attention_heads", 4),
        "attention_dropout": ckpt.get("attention_dropout", 0.1),
        "pt_k":            ckpt.get("pt_k", 16),
        "state_dict":      ckpt["model_state_dict"],
        "file_size_mb":    os.path.getsize(ckpt_path) / 1024 / 1024,
    }


def build_from_meta(meta: dict) -> torch.nn.Module:
    model = build_model(
        meta["model_type"],
        task="segmentation",
        num_classes=meta["num_classes"],
        num_features=meta["num_features"],
        k=meta["k"],
        k_small=meta["k_small"],
        k_large=meta["k_large"],
        attention_type=meta["attention_type"],
        attention_k=meta["attention_k"],
        attention_heads=meta["attention_heads"],
        attention_dropout=meta["attention_dropout"],
        pt_k=meta["pt_k"],
    )
    # Загружаем веса с strict=False — допускаем BN↔LN расхождения старых чекпоинтов.
    # Для подсчёта параметров/FLOPs это не влияет на результат.
    missing, unexpected = model.load_state_dict(meta["state_dict"], strict=False)
    if missing or unexpected:
        print(f"    (strict=False: {len(missing)} missing, {len(unexpected)} unexpected keys — OK для FLOPs/params)")
    return model


def measure_vram(model: torch.nn.Module, dummy: torch.Tensor,
                 model_type: str, dtype: torch.dtype) -> float:
    """Возвращает пиковую VRAM в МБ для одного прогона."""
    if not torch.cuda.is_available():
        return float("nan")
    device = torch.device("cuda")
    m = model.to(device)
    if dtype == torch.float16:
        m = m.half()
        x = dummy.half().to(device)
    else:
        x = dummy.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        if model_type == "pointnet":
            m(x)
        else:
            m(x)
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    m.cpu()
    del m
    torch.cuda.empty_cache()
    return peak


def count_flops(model: torch.nn.Module, dummy: torch.Tensor, model_type: str) -> float:
    """MACs в гига-единицах через thop."""
    try:
        from thop import profile as thop_profile
        model_cpu = model.cpu().eval()
        x = dummy.cpu()
        with torch.no_grad():
            macs, _ = thop_profile(model_cpu, inputs=(x,), verbose=False)
        return macs / 1e9
    except Exception as e:
        return float("nan")


def run_profiler(model: torch.nn.Module, dummy: torch.Tensor,
                 model_type: str, output_dir: Path, label: str):
    """Запускает torch.profiler, сохраняет Chrome trace и возвращает текстовую сводку."""
    if not torch.cuda.is_available():
        print(f"  [!] CUDA недоступна, пропуск профилирования {label}")
        return None

    device = torch.device("cuda")
    m = model.to(device).eval()
    x = dummy.to(device)

    trace_path = output_dir / f"{label}_trace.json"

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        # Прогрев (не записываем)
        with torch.no_grad():
            for _ in range(3):
                m(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        # Замер
        with torch.no_grad():
            for _ in range(10):
                m(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Сохраняем Chrome trace
    prof.export_chrome_trace(str(trace_path))
    print(f"  Chrome trace: {trace_path}")

    # Текстовая сводка (топ-20 по CUDA-времени)
    summary = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=20
    )
    txt_path = output_dir / f"{label}_top_ops.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== {label} — топ операций по CUDA-времени ===\n\n")
        f.write(summary)
    print(f"  Текстовая сводка: {txt_path}")

    m.cpu()
    del m
    torch.cuda.empty_cache()
    return summary


def find_best_checkpoint(checkpoints_dir: Path, model_type: str) -> Path | None:
    """Ищет best_model.pth для модели — сначала в loss_sweep, потом в корне."""
    folder_names = {
        "pointnet":          "pointnet",
        "pointnet2":         "pointnet++",
        "dgcnn":             "dgcnn",
        "ldgcnn":            "ldgcnn",
        "point_transformer": "pointtransformer",
        "ldgcnn_flash":      "ldgcnn_flash",
    }
    folder = folder_names.get(model_type, model_type)
    candidates = []
    # Предпочитаем loss_sweep (более свежие чекпоинты)
    for base in [checkpoints_dir / "loss_sweep" / folder,
                 checkpoints_dir / folder]:
        if base.exists():
            candidates += list(base.rglob("best_model.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Профилирование моделей")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints",
                        help="Папка с .pth файлами")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--output_dir", type=str, default="results/profiler")
    parser.add_argument("--no_profiler", action="store_true",
                        help="Только таблица параметров, без torch.profiler")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Запустить только для указанных моделей")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = Path(args.checkpoints_dir)
    dummy = torch.randn(1, args.num_points, 6)  # batch=1, N точек, 6 фич

    rows = []
    target_models = args.models if args.models else ORDER

    for model_type in target_models:
        if model_type not in MODEL_DISPLAY:
            print(f"Неизвестная модель: {model_type}, пропуск")
            continue

        label = MODEL_DISPLAY[model_type]
        ckpt_path = find_best_checkpoint(checkpoints_dir, model_type)

        if ckpt_path is None:
            print(f"\n[!] Чекпоинт не найден для {label}, пропуск")
            rows.append({
                "model": label, "params": "—", "gmacs": "—",
                "vram_fp32": "—", "vram_fp16": "—", "pth_mb": "—"
            })
            continue

        print(f"\n{'='*50}")
        print(f"  {label}  ({ckpt_path.name})")
        print(f"{'='*50}")

        meta = load_checkpoint_meta(str(ckpt_path))
        model = build_from_meta(meta)
        model.eval()

        # Dummy с правильным числом фич из чекпоинта
        num_feat = meta["num_features"]
        x_dummy = torch.randn(1, args.num_points, num_feat)

        # Число параметров
        n_params = sum(p.numel() for p in model.parameters())

        # FLOPs (на отдельной копии модели — thop оставляет хуки)
        print("  Считаю FLOPs...")
        gmacs = count_flops(build_from_meta(meta), x_dummy, model_type)

        # VRAM fp32
        print("  Измеряю VRAM fp32...")
        vram_fp32 = measure_vram(build_from_meta(meta), x_dummy, model_type, torch.float32)

        # VRAM fp16
        print("  Измеряю VRAM fp16...")
        vram_fp16 = measure_vram(build_from_meta(meta), x_dummy, model_type, torch.float16)

        rows.append({
            "model":    label,
            "params":   n_params,
            "gmacs":    gmacs,
            "vram_fp32": vram_fp32,
            "vram_fp16": vram_fp16,
            "pth_mb":   meta["file_size_mb"],
        })

        # torch.profiler
        if not args.no_profiler:
            print("  Запускаю torch.profiler...")
            run_profiler(model, x_dummy, model_type, output_dir, label)

    # ── Вывод расширенной таблицы ─────────────────────────────────────────────
    print("\n\n" + "="*85)
    print("ТАБЛИЦА — Параметры моделей (расширенная)")
    print("="*85)
    header = f"{'Модель':<20} {'Парам.':<12} {'GMACs':<10} {'VRAM fp32':<12} {'VRAM fp16':<12} {'Файл .pth'}"
    print(header)
    print("-"*85)

    lines = [header, "-"*85]
    for r in rows:
        def fmt(v, suffix=""):
            if isinstance(v, float) and not np.isnan(v):
                return f"{v:.1f}{suffix}"
            if isinstance(v, int):
                return f"{v:,}"
            return str(v)

        line = (f"{r['model']:<20} "
                f"{fmt(r['params']):<12} "
                f"{fmt(r['gmacs'], ' G'):<10} "
                f"{fmt(r['vram_fp32'], ' МБ'):<12} "
                f"{fmt(r['vram_fp16'], ' МБ'):<12} "
                f"{fmt(r['pth_mb'], ' МБ')}")
        print(line)
        lines.append(line)

    # Сохраняем таблицу
    table_path = output_dir / "model_stats.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nТаблица сохранена: {table_path}")


if __name__ == "__main__":
    main()
