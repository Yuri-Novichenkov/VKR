"""
Детальный анализ моделей:
  1. torchinfo  — дерево слоёв с размерами, параметрами, MFLOPs
  2. Per-layer timing — сколько мс занимает каждый именованный блок
  3. Roofline-анализ — compute-bound vs memory-bound на GPU

Использование:
    python scripts/analyze_models.py --checkpoints_dir checkpoints
    python scripts/analyze_models.py --checkpoints_dir checkpoints --models dgcnn ldgcnn_flash
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_model

# ── Константы GPU (RTX A5000) ─────────────────────────────────────────────────
# Для другого GPU передать через --gpu_tflops / --gpu_bw_gbs
GPU_PEAK_TFLOPS_FP32 = 27.77   # TFLOPs FP32
GPU_PEAK_BW_GBS      = 768.0   # GB/s memory bandwidth

MODEL_DISPLAY = {
    "pointnet":          "PointNet",
    "pointnet2":         "PointNet++",
    "dgcnn":             "DGCNN",
    "ldgcnn":            "LDGCNN",
    "point_transformer": "PointTransformer",
    "ldgcnn_flash":      "LDGCNNFlash",
}
ORDER = ["pointnet", "pointnet2", "dgcnn", "ldgcnn", "point_transformer", "ldgcnn_flash"]


# ── Загрузка чекпоинта ────────────────────────────────────────────────────────

def load_meta(ckpt_path: str) -> dict:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "model_type":        ck.get("model_type", "pointnet"),
        "num_classes":       ck["num_classes"],
        "num_features":      ck["num_features"],
        "k":                 ck.get("k", 20),
        "k_small":           ck.get("k_small", 20),
        "k_large":           ck.get("k_large", 40),
        "attention_type":    ck.get("attention_type", "none"),
        "attention_k":       ck.get("attention_k", 16),
        "attention_heads":   ck.get("attention_heads", 4),
        "attention_dropout": ck.get("attention_dropout", 0.1),
        "pt_k":              ck.get("pt_k", 16),
        "state_dict":        ck["model_state_dict"],
    }


def build(meta: dict) -> torch.nn.Module:
    m = build_model(
        meta["model_type"], task="segmentation",
        num_classes=meta["num_classes"], num_features=meta["num_features"],
        k=meta["k"], k_small=meta["k_small"], k_large=meta["k_large"],
        attention_type=meta["attention_type"], attention_k=meta["attention_k"],
        attention_heads=meta["attention_heads"], attention_dropout=meta["attention_dropout"],
        pt_k=meta["pt_k"],
    )
    m.load_state_dict(meta["state_dict"], strict=False)
    return m.eval()


def find_ckpt(checkpoints_dir: Path, model_type: str) -> Path | None:
    folder = {
        "pointnet": "pointnet", "pointnet2": "pointnet++", "dgcnn": "dgcnn",
        "ldgcnn": "ldgcnn", "point_transformer": "pointtransformer",
        "ldgcnn_flash": "ldgcnn_flash",
    }.get(model_type, model_type)
    for base in [checkpoints_dir / "loss_sweep" / folder, checkpoints_dir / folder]:
        cands = list(base.rglob("best_model.pth")) if base.exists() else []
        if cands:
            return max(cands, key=lambda p: p.stat().st_mtime)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. torchinfo
# ══════════════════════════════════════════════════════════════════════════════

def run_torchinfo(model: torch.nn.Module, meta: dict, num_points: int, output_dir: Path, label: str):
    from torchinfo import summary
    n_feat = meta["num_features"]
    result = summary(
        model,
        input_size=(1, num_points, n_feat),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        col_width=20,
        row_settings=["var_names"],
        verbose=0,
    )
    txt = str(result)
    out = output_dir / f"{label}_torchinfo.txt"
    out.write_text(f"=== {label} — torchinfo ===\n\n" + txt, encoding="utf-8")
    print(f"  torchinfo -> {out}")
    # Краткая сводка
    print(f"    Параметров:  {result.total_params:,}")
    print(f"    МАС (total): {result.total_mult_adds / 1e9:.2f} GMACs")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. Per-layer timing (CUDA events через forward hooks)
# ══════════════════════════════════════════════════════════════════════════════

def run_layer_timing(model: torch.nn.Module, meta: dict, num_points: int,
                     output_dir: Path, label: str, warmup: int = 5, reps: int = 20):
    if not torch.cuda.is_available():
        print("  [!] CUDA недоступна, пропуск per-layer timing")
        return

    device = torch.device("cuda")
    model = model.to(device)
    x = torch.randn(1, num_points, meta["num_features"], device=device)

    timings: dict[str, list[float]] = {}
    handles = []

    # Регистрируем хуки только для именованных дочерних модулей (не рекурсивно)
    for name, module in model.named_children():
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        timings[name] = []

        def make_hooks(n, s, e):
            def pre(mod, inp):
                s.record()
            def post(mod, inp, out):
                e.record()
                torch.cuda.synchronize()
                timings[n].append(s.elapsed_time(e))
            return pre, post

        pre_h, post_h = make_hooks(name, start_ev, end_ev)
        handles.append(module.register_forward_pre_hook(pre_h))
        handles.append(module.register_forward_hook(post_h))

    # Прогрев
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    # Замер
    timings = {k: [] for k in timings}  # сброс
    for name, module in model.named_children():
        for h in handles:
            h.remove()

    # Перерегистрируем для чистого замера
    handles.clear()
    for name, module in model.named_children():
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev   = torch.cuda.Event(enable_timing=True)
        timings.setdefault(name, [])

        def make_hooks(n, s, e):
            def pre(mod, inp):  s.record()
            def post(mod, inp, out):
                e.record()
                torch.cuda.synchronize()
                timings[n].append(s.elapsed_time(e))
            return pre, post

        pre_h, post_h = make_hooks(name, start_ev, end_ev)
        handles.append(module.register_forward_pre_hook(pre_h))
        handles.append(module.register_forward_hook(post_h))

    with torch.no_grad():
        for _ in range(reps):
            model(x)

    for h in handles:
        h.remove()

    # Считаем среднее
    avg = {k: float(np.mean(v)) for k, v in timings.items() if v}
    total = sum(avg.values())

    lines = [f"=== {label} — per-layer timing (среднее по {reps} прогонам) ===\n",
             f"{'Блок':<35} {'мс':>8}  {'%':>6}", "-" * 55]
    for name, ms in sorted(avg.items(), key=lambda x: -x[1]):
        pct = ms / total * 100 if total else 0
        lines.append(f"{name:<35} {ms:>8.3f}  {pct:>5.1f}%")
    lines.append("-" * 55)
    lines.append(f"{'ИТОГО (сумма слоёв)':<35} {total:>8.3f}")

    out = output_dir / f"{label}_layer_timing.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  per-layer timing -> {out}")
    # Вывод топ-3
    top3 = sorted(avg.items(), key=lambda x: -x[1])[:3]
    for name, ms in top3:
        print(f"    {name:<30} {ms:.3f} мс  ({ms/total*100:.1f}%)")

    model.cpu()
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# 3. Roofline-анализ
# ══════════════════════════════════════════════════════════════════════════════

def compute_roofline_point(meta: dict, model: torch.nn.Module,
                           num_points: int, device: torch.device) -> tuple[float, float]:
    """
    Возвращает (arithmetic_intensity, achieved_gflops).
    AI = GFLOPs / GB_accessed
    """
    from thop import profile as thop_profile

    n_feat = meta["num_features"]
    x_cpu  = torch.randn(1, num_points, n_feat)

    # FLOPs
    m_cpu = build(meta).eval()
    with torch.no_grad():
        macs, _ = thop_profile(m_cpu, inputs=(x_cpu,), verbose=False)
    gflops = macs * 2 / 1e9  # MACs -> FLOPs (умножение + сложение)

    # Bytes accessed = параметры (fp32) + активации (peak VRAM на прогон)
    param_bytes = sum(p.numel() * 4 for p in m_cpu.parameters())

    if torch.cuda.is_available():
        m_gpu = build(meta).to(device).eval()
        x_gpu = torch.randn(1, num_points, n_feat, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            m_gpu(x_gpu)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        m_gpu.cpu()
        del m_gpu
        torch.cuda.empty_cache()
        # Активации ≈ peak − веса (приближение)
        act_bytes = max(peak_bytes - param_bytes, 0)
        total_bytes = param_bytes + act_bytes
    else:
        total_bytes = param_bytes * 3  # грубая оценка без GPU

    gb_accessed = total_bytes / 1e9
    ai = gflops / gb_accessed if gb_accessed > 0 else 0

    # Достигнутый GFLOPS = FLOPs / время
    if torch.cuda.is_available():
        m_gpu = build(meta).to(device).eval()
        x_gpu = torch.randn(1, num_points, n_feat, device=device)
        with torch.no_grad():
            for _ in range(3):  # прогрев
                m_gpu(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                m_gpu(x_gpu)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 20
        achieved_gflops = gflops / elapsed
        m_gpu.cpu()
        del m_gpu
        torch.cuda.empty_cache()
    else:
        achieved_gflops = float("nan")

    return ai, achieved_gflops, gflops, gb_accessed


def run_roofline(models_data: list[dict], output_dir: Path,
                 peak_tflops: float, peak_bw_gbs: float):
    """
    Строит Roofline-график и сохраняет PNG + текстовую таблицу.
    models_data: список {"label", "ai", "achieved_gflops", "gflops", "gb"}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Roofline: y = min(peak_flops, AI * peak_bw)
    peak_gflops = peak_tflops * 1000  # TFLOPs -> GFLOPs
    ridge_ai = peak_gflops / (peak_bw_gbs * 1000 / 1)  # FLOPs/byte -> нормировка
    # ridge_ai: при peak_bw в GB/s, peak_gflops в GFLOPs
    # AI в GFLOPs/GB = FLOPs/byte
    ridge_point = peak_gflops / (peak_bw_gbs)  # GFLOPs/GB

    ai_range = np.logspace(-2, 4, 500)
    roof = np.minimum(peak_gflops, ai_range * peak_bw_gbs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.loglog(ai_range, roof, "k-", linewidth=2, label="Roofline (теоретический максимум)")
    ax.axvline(ridge_point, color="gray", linestyle="--", alpha=0.5,
               label=f"Ridge point ({ridge_point:.1f} FLOPs/byte)")

    colors = plt.get_cmap("tab10", len(models_data))
    for i, d in enumerate(models_data):
        if not np.isnan(d["achieved_gflops"]):
            ax.scatter(d["ai"], d["achieved_gflops"],
                       color=colors(i), s=120, zorder=5,
                       label=f"{d['label']}  (AI={d['ai']:.1f}, {d['achieved_gflops']:.0f} GFLOPs)")
            # Вертикальная линия до потолка
            roof_at_ai = min(peak_gflops, d["ai"] * peak_bw_gbs)
            ax.plot([d["ai"], d["ai"]], [d["achieved_gflops"], roof_at_ai],
                    color=colors(i), linestyle=":", alpha=0.5)

    ax.set_xlabel("Arithmetic Intensity (GFLOPs / GB)", fontsize=12)
    ax.set_ylabel("Performance (GFLOPs/s)", fontsize=12)
    ax.set_title("Roofline-анализ моделей сегментации (RTX A5000, fp32)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(1, peak_gflops * 2)

    # Подпись зон
    ax.text(0.015, peak_bw_gbs * 0.015 * 1.5, "Memory-bound ->", fontsize=9,
            color="gray", rotation=45, va="bottom")
    ax.text(ridge_point * 1.1, peak_gflops * 0.5, "← Compute-bound",
            fontsize=9, color="gray", va="center")

    plt.tight_layout()
    png_path = output_dir / "roofline.png"
    plt.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Roofline график -> {png_path}")

    # Текстовая таблица
    lines = [
        f"=== Roofline-анализ (GPU: {peak_tflops} TFLOPs fp32, BW: {peak_bw_gbs} GB/s) ===\n",
        f"Ridge point: {ridge_point:.1f} GFLOPs/GB  (выше -> compute-bound, ниже -> memory-bound)\n",
        f"{'Модель':<20} {'GFLOPs':>8} {'GB accs':>8} {'AI (F/B)':>10} {'GFLOPs/s':>10} {'Потолок':>10} {'Утилиз.':>8} {'Режим'}",
        "-" * 90,
    ]
    for d in models_data:
        roof_at_ai = min(peak_gflops, d["ai"] * peak_bw_gbs)
        util = d["achieved_gflops"] / roof_at_ai * 100 if not np.isnan(d["achieved_gflops"]) else float("nan")
        mode = "compute-bound" if d["ai"] > ridge_point else "memory-bound"
        lines.append(
            f"{d['label']:<20} {d['gflops']:>8.1f} {d['gb']:>8.2f} "
            f"{d['ai']:>10.2f} {d['achieved_gflops']:>10.0f} "
            f"{roof_at_ai:>10.0f} {util:>7.1f}% {mode}"
        )

    txt_path = output_dir / "roofline_table.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Roofline таблица -> {txt_path}")


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--num_points", type=int, default=4096)
    parser.add_argument("--output_dir", default="results/profiler")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--gpu_tflops", type=float, default=GPU_PEAK_TFLOPS_FP32)
    parser.add_argument("--gpu_bw_gbs", type=float, default=GPU_PEAK_BW_GBS)
    parser.add_argument("--skip_torchinfo", action="store_true")
    parser.add_argument("--skip_timing", action="store_true")
    parser.add_argument("--skip_roofline", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = args.models or ORDER

    roofline_data = []

    for model_type in target:
        if model_type not in MODEL_DISPLAY:
            continue
        label = MODEL_DISPLAY[model_type]
        ckpt = find_ckpt(ckpt_dir, model_type)
        if ckpt is None:
            print(f"\n[!] Чекпоинт не найден: {label}")
            continue

        print(f"\n{'='*55}\n  {label}\n{'='*55}")
        meta = load_meta(str(ckpt))

        # 1. torchinfo
        if not args.skip_torchinfo:
            model = build(meta)
            run_torchinfo(model, meta, args.num_points, output_dir, label)

        # 2. Per-layer timing
        if not args.skip_timing:
            model = build(meta)
            run_layer_timing(model, meta, args.num_points, output_dir, label)

        # 3. Roofline (сбор данных)
        if not args.skip_roofline:
            print("  Считаю roofline точку...")
            ai, ach_gflops, gflops, gb = compute_roofline_point(
                meta, build(meta), args.num_points, device
            )
            print(f"    AI={ai:.2f} GFLOPs/GB  achieved={ach_gflops:.0f} GFLOPs/s")
            roofline_data.append({
                "label": label, "ai": ai,
                "achieved_gflops": ach_gflops,
                "gflops": gflops, "gb": gb,
            })

    # Строим Roofline для всех моделей сразу
    if not args.skip_roofline and roofline_data:
        run_roofline(roofline_data, output_dir, args.gpu_tflops, args.gpu_bw_gbs)


if __name__ == "__main__":
    main()
