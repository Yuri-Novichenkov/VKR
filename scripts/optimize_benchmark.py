"""
Бенчмарк оптимизаций:
  1. fast_knn=True  — torch_cluster вместо O(N²) матрицы расстояний
  2. INT8            — dynamic quantization (torch.ao.quantization)

Для каждой модели измеряем:
  - Время инференса FP32 (baseline)
  - Время инференса FP32 + fast_knn
  - Время инференса FP16 + fast_knn
  - Время инференса INT8 (dynamic quant, CPU или GPU)
  - Ускорение относительно baseline

Сохраняет: results/optimization_results.xlsx

Запуск:
    python scripts/optimize_benchmark.py
"""

import sys, torch, numpy as np
from pathlib import Path

sys.path.insert(0, ".")
from src.models import build_model

# Проверяем доступность torch_cluster
try:
    import torch_cluster  # noqa: F401
    HAS_TORCH_CLUSTER = True
except ImportError:
    HAS_TORCH_CLUSTER = False
    print("WARNING: torch_cluster не установлен. fast_knn будет пропущен.")
    print("Установка: pip install torch-cluster -f https://data.pyg.org/whl/torch-<ver>+<cuda>.html")

CONFIGS = [
    ("DGCNN",            "dgcnn",            "checkpoints/loss_sweep/dgcnn/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN",           "ldgcnn",           "checkpoints/loss_sweep/ldgcnn/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN GAT",       "ldgcnn",           "checkpoints/loss_sweep/ldgcnn/segmentation/attn_gatv2_k16_h4_d0p1__cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN WINDOW",    "ldgcnn",           "checkpoints/loss_sweep/ldgcnn/segmentation/attn_local_window_k16_h4_d0p1__cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNNFlash",      "ldgcnn_flash",     "checkpoints/loss_sweep/ldgcnn_flash/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("PointTransformer", "pointtransformer", "checkpoints/pointtransformer/segmentation/cb_effective_b0p999/mar16/best_model.pth"),
]
DEFAULTS = dict(k=20, k_small=20, k_large=40, attention_type="none",
                attention_k=16, attention_heads=4, attention_dropout=0.1, pt_k=16)
N_POINTS = 4096
WARMUP, REPS = 5, 30


def load_model(path, fast_knn=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = {**DEFAULTS}
    for key in DEFAULTS:
        if ck.get(key) is not None:
            meta[key] = ck[key]
    mtype = ck.get("model_type", "pointnet")
    m = build_model(
        mtype, task="segmentation",
        num_classes=ck["num_classes"], num_features=ck["num_features"],
        k=meta["k"], k_small=meta["k_small"], k_large=meta["k_large"],
        attention_type=meta["attention_type"], attention_k=meta["attention_k"],
        attention_heads=meta["attention_heads"], attention_dropout=meta["attention_dropout"],
        pt_k=meta["pt_k"],
        fast_knn=fast_knn,
    )
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.eval(), ck["num_features"]


def bench_gpu(model, num_features, device, dtype=torch.float32):
    """Возвращает среднее время инференса (мс) на GPU через CUDA events."""
    model = model.to(device)
    if dtype == torch.float16:
        model = model.half()
    x = torch.randn(1, N_POINTS, num_features, device=device, dtype=dtype)
    # Прогрев
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
    torch.cuda.synchronize()
    # Замер
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(REPS):
            s.record()
            model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    model.cpu()
    torch.cuda.empty_cache()
    return float(np.mean(times))


def bench_cpu(model, num_features):
    """Среднее время инференса на CPU (мс)."""
    x = torch.randn(1, N_POINTS, num_features)
    with torch.no_grad():
        for _ in range(3):
            model(x)
    times = []
    with torch.no_grad():
        import time
        for _ in range(10):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def apply_int8(model):
    """Dynamic INT8 quantization: квантует Linear, Conv1d и Conv2d слои.
    Conv2d критичен для EdgeConv-based моделей (DGCNN/LDGCNN/LDGCNNFlash).
    """
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d},
        dtype=torch.qint8,
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = []

for label, mtype, path in CONFIGS:
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    # ── FP32 baseline ─────────────────────────────────────────────────────
    model, num_feat = load_model(path, fast_knn=False)
    t_fp32 = bench_gpu(model, num_feat, device, torch.float32)
    print(f"  FP32 baseline:       {t_fp32:.2f} ms")

    # ── FP32 + fast_knn ───────────────────────────────────────────────────
    t_fp32_fk = float("nan")
    t_fp16_fk = float("nan")
    if HAS_TORCH_CLUSTER:
        model_fk, _ = load_model(path, fast_knn=True)
        t_fp32_fk = bench_gpu(model_fk, num_feat, device, torch.float32)
        print(f"  FP32 + fast_knn:     {t_fp32_fk:.2f} ms  (x{t_fp32/t_fp32_fk:.2f})")

        model_fk16, _ = load_model(path, fast_knn=True)
        t_fp16_fk = bench_gpu(model_fk16, num_feat, device, torch.float16)
        print(f"  FP16 + fast_knn:     {t_fp16_fk:.2f} ms  (x{t_fp32/t_fp16_fk:.2f})")
    else:
        print("  fast_knn: пропущено (torch_cluster не установлен)")

    # ── INT8 dynamic (CPU) ────────────────────────────────────────────────
    model_int8, _ = load_model(path, fast_knn=False)
    model_int8 = apply_int8(model_int8)
    t_int8 = bench_cpu(model_int8, num_feat)
    model_fp32_cpu, _ = load_model(path, fast_knn=False)
    t_fp32_cpu = bench_cpu(model_fp32_cpu, num_feat)
    print(f"  INT8 (CPU):          {t_int8:.1f} ms  (x{t_fp32_cpu/t_int8:.2f} vs FP32 CPU)")

    def _x(a, b):
        return round(a / b, 2) if b and not (isinstance(b, float) and b != b) else None

    results.append({
        "model":       label,
        "fp32_ms":     round(t_fp32, 2),
        "fp32_fk_ms":  round(t_fp32_fk, 2) if HAS_TORCH_CLUSTER else "—",
        "fp32_fk_x":   _x(t_fp32, t_fp32_fk) if HAS_TORCH_CLUSTER else "—",
        "fp16_fk_ms":  round(t_fp16_fk, 2) if HAS_TORCH_CLUSTER else "—",
        "fp16_fk_x":   _x(t_fp32, t_fp16_fk) if HAS_TORCH_CLUSTER else "—",
        "int8_cpu_ms": round(t_int8, 1),
        "fp32_cpu_ms": round(t_fp32_cpu, 1),
        "int8_x":      _x(t_fp32_cpu, t_int8),
    })

# ── Консольная таблица ────────────────────────────────────────────────────────
def _fmt(val, fmt):
    """Форматирует число или возвращает '—' для строк/None/nan."""
    if val is None:
        return "—"
    if isinstance(val, str):
        return val
    try:
        if val != val:  # nan
            return "—"
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)

print("\n\n" + "="*90)
print(f"{'Модель':<16} {'FP32':>8} {'FP32+fkNN':>10} {'x':>5} {'FP16+fkNN':>10} {'x':>5} {'INT8(CPU)':>10} {'x':>5}")
print("-"*90)
for r in results:
    print(f"{r['model']:<16} {_fmt(r['fp32_ms'],'.2f'):>8}"
          f" {_fmt(r['fp32_fk_ms'],'.2f'):>10} {_fmt(r['fp32_fk_x'],'.2f'):>5}"
          f" {_fmt(r['fp16_fk_ms'],'.2f'):>10} {_fmt(r['fp16_fk_x'],'.2f'):>5}"
          f" {_fmt(r['int8_cpu_ms'],'.1f'):>10} {_fmt(r['int8_x'],'.2f'):>5}")

# ── Excel ─────────────────────────────────────────────────────────────────────
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference

wb = Workbook()
ws = wb.active
ws.title = "Оптимизация kNN + INT8"

HDR   = PatternFill("solid", start_color="BDD7EE")
GREEN = PatternFill("solid", start_color="C6EFCE")
BLUE  = PatternFill("solid", start_color="DDEEFF")
thin  = Side(style="thin", color="AAAAAA")
brd   = Border(left=thin, right=thin, top=thin, bottom=thin)
MODEL_COLORS = ["DEEAF1","E2EFDA","FFF2CC","FCE4D6","EAE1F4","D9EAD3"]
mc = {r["model"]: MODEL_COLORS[i] for i, r in enumerate(results)}

headers = [
    "Модель",
    "FP32 (мс)\nbaseline GPU",
    "FP32 + fast kNN (мс)",
    "Ускорение kNN (x)",
    "FP16 + fast kNN (мс)",
    "Ускорение FP16+kNN (x)",
    "INT8 CPU (мс)",
    "FP32 CPU (мс)",
    "Ускорение INT8 (x)",
]
ws.append(headers)
for col, h in enumerate(headers, 1):
    c = ws.cell(1, col)
    c.value = h
    c.font = Font(bold=True, name="Arial", size=9)
    c.fill = HDR
    c.alignment = Alignment(horizontal="center", wrap_text=True)
    c.border = brd

for r_idx, r in enumerate(results, 2):
    fill = PatternFill("solid", start_color=mc[r["model"]])
    vals = [r["model"], r["fp32_ms"], r["fp32_fk_ms"], r["fp32_fk_x"],
            r["fp16_fk_ms"], r["fp16_fk_x"], r["int8_cpu_ms"], r["fp32_cpu_ms"], r["int8_x"]]
    for col, val in enumerate(vals, 1):
        c = ws.cell(r_idx, col)
        c.value = val
        c.font = Font(name="Arial", size=10)
        c.fill = fill
        c.border = brd
        c.alignment = Alignment(horizontal="center")
    # Подсветить столбцы ускорения зелёным если > 1.3
    for col, key in [(4, "fp32_fk_x"), (6, "fp16_fk_x"), (9, "int8_x")]:
        if r[key] >= 1.5:
            ws.cell(r_idx, col).fill = GREEN
            ws.cell(r_idx, col).font = Font(bold=True, name="Arial", size=10)

col_widths = [16, 14, 20, 17, 20, 22, 13, 13, 17]
for i, w in enumerate(col_widths, 1):
    ws.column_dimensions[ws.cell(1, i).column_letter].width = w
for r in range(1, len(results) + 2):
    ws.row_dimensions[r].height = 32

# График — ускорения
ws2 = wb.create_sheet("Графики ускорения")
ws2.append(["Модель", "FP32+fkNN (x)", "FP16+fkNN (x)", "INT8 CPU (x)"])
for r in results:
    ws2.append([r["model"], r["fp32_fk_x"], r["fp16_fk_x"], r["int8_x"]])

chart = BarChart()
chart.type = "col"
chart.title = "Ускорение относительно FP32 baseline"
chart.y_axis.title = "Ускорение (x)"
chart.style = 10
chart.width = 24
chart.height = 14
data = Reference(ws2, min_col=2, max_col=4, min_row=1, max_row=len(results) + 1)
cats = Reference(ws2, min_col=1, min_row=2, max_row=len(results) + 1)
chart.add_data(data, titles_from_data=True)
chart.set_categories(cats)
ws2.add_chart(chart, "F2")

# Лист 3 — абсолютные времена
ws3 = wb.create_sheet("Абсолютные времена")
ws3.append(["Модель", "FP32 GPU (мс)", "FP32+fkNN GPU (мс)", "FP16+fkNN GPU (мс)", "INT8 CPU (мс)"])
for r in results:
    ws3.append([r["model"], r["fp32_ms"], r["fp32_fk_ms"], r["fp16_fk_ms"], r["int8_cpu_ms"]])

chart2 = BarChart()
chart2.type = "col"
chart2.title = "Время инференса по режимам (мс)"
chart2.y_axis.title = "мс"
chart2.style = 10
chart2.width = 24
chart2.height = 14
data2 = Reference(ws3, min_col=2, max_col=5, min_row=1, max_row=len(results) + 1)
cats2 = Reference(ws3, min_col=1, min_row=2, max_row=len(results) + 1)
chart2.add_data(data2, titles_from_data=True)
chart2.set_categories(cats2)
ws3.add_chart(chart2, "F2")

out = "results/optimization_results.xlsx"
wb.save(out)
print(f"\nsaved: {out}")
