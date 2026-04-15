"""
Export project models to Netron-friendly formats.

By default this script creates TorchScript traced models that can be opened
in Netron as a technical complement to the thesis diagrams.
"""

import argparse
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.dgcnn import DGCNNSegmentation
from src.models.ldgcnn import LDGCNNSegmentation
from src.models.pointnet import PointNetSegmentation
from src.models.pointnet_plusplus import PointNetPlusPlusSegmentation


def build_model(args, model_name: str):
    model_name = model_name.lower()
    if model_name == "pointnet":
        return PointNetSegmentation(
            num_classes=args.num_classes,
            num_features=args.num_features,
        )
    if model_name == "pointnetpp":
        return PointNetPlusPlusSegmentation(
            num_classes=args.num_classes,
            num_features=args.num_features,
        )
    if model_name == "dgcnn":
        return DGCNNSegmentation(
            num_classes=args.num_classes,
            num_features=args.num_features,
            k=args.k,
        )
    if model_name == "ldgcnn":
        return LDGCNNSegmentation(
            num_classes=args.num_classes,
            num_features=args.num_features,
            k_small=args.k_small,
            k_large=args.k_large,
            attention_type=args.attention_type,
            attention_k=args.attention_k,
            attention_heads=args.attention_heads,
            attention_dropout=args.attention_dropout,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_filename(args, model_name: str) -> str:
    base = f"{model_name}_segmentation_np{args.num_points}_nf{args.num_features}_nc{args.num_classes}"
    if model_name == "dgcnn":
        base += f"_k{args.k}"
    elif model_name == "ldgcnn":
        base += (
            f"_ks{args.k_small}_kl{args.k_large}"
            f"_attn-{args.attention_type}"
            f"_ak{args.attention_k}_ah{args.attention_heads}"
        )
    return base


def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, output_path: Path) -> None:
    traced = torch.jit.trace(model, example_input, strict=False, check_trace=False)
    rel_output = Path(os.path.relpath(output_path, ROOT)).as_posix()
    traced.save(rel_output)


def main():
    parser = argparse.ArgumentParser(description="Export segmentation models for Netron")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["pointnet", "pointnetpp", "dgcnn", "ldgcnn"],
        help="Models to export: pointnet, pointnetpp, dgcnn, ldgcnn",
    )
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--num_features", type=int, default=9)
    parser.add_argument("--num_classes", type=int, default=11)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--k_small", type=int, default=20)
    parser.add_argument("--k_large", type=int, default=40)
    parser.add_argument("--attention_type", type=str, default="gatv2")
    parser.add_argument("--attention_k", type=int, default=16)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diagrams/netron",
        help="Directory for exported Netron files",
    )
    args = parser.parse_args()

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    example_input = torch.randn(1, args.num_points, args.num_features)

    for model_name in args.models:
        try:
            model = build_model(args, model_name)
            model.eval()
            output_name = build_filename(args, model_name) + ".pt"
            output_path = out_dir / output_name
            export_torchscript(model, example_input, output_path)
            print(f"Saved: {output_path}")
        except Exception as exc:
            print(f"FAILED: {model_name}: {exc}")


if __name__ == "__main__":
    main()
