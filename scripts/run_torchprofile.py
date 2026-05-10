import torch, sys
sys.path.insert(0, ".")
from torchprofile import profile_macs
from src.models import build_model

configs = [
    ("PointNet",         "pointnet",          "checkpoints/loss_sweep/pointnet/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("PointNet++",       "pointnet++",        "checkpoints/loss_sweep/pointnet++/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("DGCNN",            "dgcnn",             "checkpoints/loss_sweep/dgcnn/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN",           "ldgcnn",            "checkpoints/loss_sweep/ldgcnn/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN GAT",       "ldgcnn",            "checkpoints/loss_sweep/ldgcnn/segmentation/attn_gatv2_k16_h4_d0p1__cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNN WINDOW",    "ldgcnn",            "checkpoints/loss_sweep/ldgcnn/segmentation/attn_local_window_k16_h4_d0p1__cb_effective_b0p99999/mar16/best_model.pth"),
    ("LDGCNNFlash",      "ldgcnn_flash",      "checkpoints/loss_sweep/ldgcnn_flash/segmentation/cb_effective_b0p99999/mar16/best_model.pth"),
    ("PointTransformer", "pointtransformer",  "checkpoints/pointtransformer/segmentation/cb_effective_b0p999/mar16/best_model.pth"),
]

defaults = dict(k=20, k_small=20, k_large=40, attention_type="none",
                attention_k=16, attention_heads=4, attention_dropout=0.1, pt_k=16)

results = []
for label, mtype, path in configs:
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = {**defaults}
        for key in defaults:
            if ck.get(key) is not None:
                meta[key] = ck[key]
        m = build_model(
            mtype, task="segmentation",
            num_classes=ck["num_classes"], num_features=ck["num_features"],
            k=meta["k"], k_small=meta["k_small"], k_large=meta["k_large"],
            attention_type=meta["attention_type"], attention_k=meta["attention_k"],
            attention_heads=meta["attention_heads"], attention_dropout=meta["attention_dropout"],
            pt_k=meta["pt_k"],
        )
        m.load_state_dict(ck["model_state_dict"], strict=False)
        m.eval()
        x = torch.randn(1, 4096, ck["num_features"])
        macs = profile_macs(m, x)
        params = sum(p.numel() for p in m.parameters())
        results.append((label, macs, params))
        print("OK: " + label)
    except Exception as e:
        results.append((label, None, None))
        print("ERR " + label + ": " + str(e))

print("\n=== torchprofile результаты ===")
print("{:<20} {:>10} {:>14}".format("Модель", "GMACs", "Параметров"))
print("-" * 48)
for label, macs, params in results:
    if macs is not None:
        print("{:<20} {:>10.2f} {:>14,}".format(label, macs / 1e9, params))
    else:
        print("{:<20} {:>10} {:>14}".format(label, "---", "---"))

with open("results/torchprofile_results.txt", "w", encoding="utf-8") as f:
    f.write("Модель               GMACs       Параметров\n")
    f.write("-" * 48 + "\n")
    for label, macs, params in results:
        if macs is not None:
            f.write("{:<20} {:>10.2f} {:>14,}\n".format(label, macs / 1e9, params))
        else:
            f.write("{:<20} {:>10} {:>14}\n".format(label, "---", "---"))
print("saved: results/torchprofile_results.txt")
