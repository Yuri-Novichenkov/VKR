"""
Рисунок 3.1 — Диаграмма значений mIoU по моделям
(Mar16, тестовая выборка со сценовым голосованием)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Данные (test.py + scene-level voting, Mar16) ──────────────────────────
MODELS = [
    ("PointNet",          35.62),
    ("PointNet++",        44.69),
    ("DGCNN",             55.45),
    ("LDGCNN",            53.75),
    ("LDGCNN-GATv2",      55.05),
    ("LDGCNN-LocalWin",   55.97),
    ("LDGCNNFlash",       60.88),
    ("PointTransformer",  61.65),
]

# Сортировка по mIoU
MODELS = sorted(MODELS, key=lambda x: x[1])
names  = [m[0] for m in MODELS]
values = [m[1] for m in MODELS]

# ── Цвета: семейство → оттенок ─────────────────────────────────────────────
FAMILY_COLOR = {
    "PointNet":          "#5B9BD5",   # синий
    "PointNet++":        "#5B9BD5",
    "DGCNN":             "#70AD47",   # зелёный
    "LDGCNN":            "#70AD47",
    "LDGCNN-GATv2":      "#70AD47",
    "LDGCNN-LocalWin":   "#70AD47",
    "LDGCNNFlash":       "#ED7D31",   # оранжевый (наша модель)
    "PointTransformer":  "#C00000",   # тёмно-красный (лучший)
}
colors = [FAMILY_COLOR[n] for n in names]

# ── Размер фигуры (под ВКР: ≈ ширина 16 см) ───────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))

y_pos = np.arange(len(names))
bars  = ax.barh(y_pos, values, color=colors, edgecolor="white",
                linewidth=0.6, height=0.6)

# Подписи значений
for bar, val in zip(bars, values):
    ax.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center", ha="left",
        fontsize=9.5, color="#222222",
        fontweight="bold",
    )

# Оси
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("mIoU, %", fontsize=10)
ax.set_xlim(0, max(values) + 8)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.tick_params(axis="x", labelsize=9)

# Вертикальная сетка
ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6, color="gray")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

# Легенда семейств
legend_handles = [
    mpatches.Patch(color="#5B9BD5", label="PointNet-семейство"),
    mpatches.Patch(color="#70AD47", label="DGCNN/LDGCNN-семейство"),
    mpatches.Patch(color="#ED7D31", label="LDGCNNFlash (предлагаемый)"),
    mpatches.Patch(color="#C00000", label="PointTransformer"),
]
ax.legend(handles=legend_handles, fontsize=8.5,
          loc="center left", bbox_to_anchor=(1.01, 0.5),
          framealpha=0.9, edgecolor="#cccccc")

plt.tight_layout()

out = Path("results/fig_miou_bar_mar16.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved: {out}")
