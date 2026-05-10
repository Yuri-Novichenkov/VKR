"""
Отладочный скрипт v5 — финальная проверка после фикса _square_distance.
Запуск: ./.venv/bin/python scripts/debug_pt_nan.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch import amp

from src.models.point_transformer import (
    PointTransformerSegmentation,
    _knn_query, _index_points, _apply_bn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device},  torch: {torch.__version__}")

torch.manual_seed(42)
B, N, nf, C = 2, 4096, 9, 11
x = torch.randn(B, N, nf, device=device)
labels = torch.randint(0, C, (B, N), device=device)

model = PointTransformerSegmentation(num_classes=C, num_features=nf, k=16).to(device)
model.train()


def chk(name, t):
    ok = torch.isfinite(t).all()
    rng = f"[{t.float().min():.4f}, {t.float().max():.4f}]"
    status = "OK" if ok else f"*** NaN/Inf: {(~torch.isfinite(t)).sum()}/{t.numel()} ***"
    print(f"  {name:45s} {status:40s} dtype={t.dtype}  shape={tuple(t.shape)}  {rng}")
    return not ok


print("\n=== Full forward pass под AMP ===")
with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
    pos = x[:, :, :3].contiguous()

    # STEM
    x0 = F.relu(
        _apply_bn(model.stem_fc(x.reshape(B * N, nf)), model.stem_bn),
        inplace=False,
    ).reshape(B, N, -1)
    if chk("x0 (stem)", x0): sys.exit(1)

    # ENCODER
    idx0 = _knn_query(model.enc0.k, pos, pos)
    x0 = x0 + model.enc0.pt(_apply_bn(x0, model.enc0.bn), pos, idx0)
    if chk("x0 (enc0)", x0): sys.exit(1)

    x1, pos1, idx1 = model.td1(x0, pos)
    if chk("x1 (td1)", x1): sys.exit(1)
    x1 = x1 + model.enc1.pt(_apply_bn(x1, model.enc1.bn), pos1, idx1)
    if chk("x1 (enc1)", x1): sys.exit(1)

    x2, pos2, idx2 = model.td2(x1, pos1)
    if chk("x2 (td2)", x2): sys.exit(1)
    x2 = x2 + model.enc2.pt(_apply_bn(x2, model.enc2.bn), pos2, idx2)
    if chk("x2 (enc2)", x2): sys.exit(1)

    x3, pos3, idx3 = model.td3(x2, pos2)
    if chk("x3 (td3)", x3): sys.exit(1)
    x3 = x3 + model.enc3.pt(_apply_bn(x3, model.enc3.bn), pos3, idx3)
    if chk("x3 (enc3)", x3): sys.exit(1)

    x4, pos4, idx4 = model.td4(x3, pos3)
    if chk("x4 (td4)", x4): sys.exit(1)
    x4 = x4 + model.enc4.pt(_apply_bn(x4, model.enc4.bn), pos4, idx4)
    if chk("x4 (enc4)", x4): sys.exit(1)

    print("  --- DECODER ---")

    # DECODER
    d3 = model.tu3(x4, pos4, x3, pos3)
    if chk("d3 (tu3)", d3): sys.exit(1)
    idx = _knn_query(model.dec3.k, pos3, pos3)
    d3 = d3 + model.dec3.pt(_apply_bn(d3, model.dec3.bn), pos3, idx)
    if chk("d3 (dec3)", d3): sys.exit(1)

    d2 = model.tu2(d3, pos3, x2, pos2)
    if chk("d2 (tu2)", d2): sys.exit(1)
    idx = _knn_query(model.dec2.k, pos2, pos2)
    d2 = d2 + model.dec2.pt(_apply_bn(d2, model.dec2.bn), pos2, idx)
    if chk("d2 (dec2)", d2): sys.exit(1)

    d1 = model.tu1(d2, pos2, x1, pos1)
    if chk("d1 (tu1)", d1): sys.exit(1)
    idx = _knn_query(model.dec1.k, pos1, pos1)
    d1 = d1 + model.dec1.pt(_apply_bn(d1, model.dec1.bn), pos1, idx)
    if chk("d1 (dec1)", d1): sys.exit(1)

    d0 = model.tu0(d1, pos1, x0, pos)
    if chk("d0 (tu0)", d0): sys.exit(1)
    idx = _knn_query(model.dec0.k, pos, pos)
    d0 = d0 + model.dec0.pt(_apply_bn(d0, model.dec0.bn), pos, idx)
    if chk("d0 (dec0)", d0): sys.exit(1)

    out = model.head(d0)
    if chk("out (head)", out): sys.exit(1)

print("\n=== Backward pass ===")
# Проверяем что backward тоже не даёт NaN
scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()

with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
    out2 = model(x)
    loss = F.cross_entropy(out2.reshape(-1, C), labels.reshape(-1))

print(f"  loss = {loss.item():.6f}  {'OK' if torch.isfinite(loss) else '*** NaN/Inf ***'}")
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Проверяем градиенты
nan_grads = 0
for name_p, p in model.named_parameters():
    if p.grad is not None and not torch.isfinite(p.grad).all():
        nan_grads += 1
        print(f"  *** NaN grad: {name_p}")
if nan_grads == 0:
    print("  Все градиенты конечны — OK")
else:
    print(f"  Итого параметров с NaN градиентом: {nan_grads}")

print("\nДиагностика завершена.")
