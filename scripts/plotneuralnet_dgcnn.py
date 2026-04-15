"""
Generate a DGCNN (segmentation) architecture diagram using PlotNeuralNet.

Output: TeX file that can be compiled with LaTeX/TikZ.
"""

import argparse
import os
import sys
from pathlib import Path


def build_arch(tools_path: Path):
    from pycore.tikzeng import to_Conv, to_Pool, to_begin, to_connection, to_cor, to_end, to_generate, to_head

    arch = [
        to_head(str(tools_path)),
        to_cor(),
        to_begin(),
        to_Conv(
            "input",
            s_filer="",
            n_filer="",
            offset="(0,0,0)",
            to="(0,0,0)",
            width=1.8,
            height=36,
            depth=36,
            caption=r"Input\\{\scriptsize 9x2048}",
        ),
        to_Pool(
            "knn",
            offset="(2.8,0,0)",
            to="(input-east)",
            width=1.2,
            height=32,
            depth=32,
            caption=r"kNN\\{\scriptsize k=20}",
        ),
        to_connection("input", "knn"),
        to_Conv(
            "ec1",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(knn-east)",
            width=1.8,
            height=30,
            depth=30,
            caption=r"EC1\\{\scriptsize 64x2048}",
        ),
        to_connection("knn", "ec1"),
        to_Pool(
            "knn2",
            offset="(2.4,0,0)",
            to="(ec1-east)",
            width=1.0,
            height=28,
            depth=28,
            caption=r"DynkNN\\{\scriptsize k=20}",
        ),
        to_connection("ec1", "knn2"),
        to_Conv(
            "ec2",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(knn2-east)",
            width=1.8,
            height=28,
            depth=28,
            caption=r"EC2\\{\scriptsize 64x2048}",
        ),
        to_connection("knn2", "ec2"),
        to_Pool(
            "knn3",
            offset="(2.4,0,0)",
            to="(ec2-east)",
            width=1.0,
            height=26,
            depth=26,
            caption=r"DynkNN\\{\scriptsize k=20}",
        ),
        to_connection("ec2", "knn3"),
        to_Conv(
            "ec3",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(knn3-east)",
            width=1.8,
            height=26,
            depth=26,
            caption=r"EC3\\{\scriptsize 128x2048}",
        ),
        to_connection("knn3", "ec3"),
        to_Pool(
            "knn4",
            offset="(2.4,0,0)",
            to="(ec3-east)",
            width=1.0,
            height=24,
            depth=24,
            caption=r"DynkNN\\{\scriptsize k=20}",
        ),
        to_connection("ec3", "knn4"),
        to_Conv(
            "ec4",
            s_filer="",
            n_filer="",
            offset="(2.4,0,0)",
            to="(knn4-east)",
            width=1.8,
            height=24,
            depth=24,
            caption=r"EC4\\{\scriptsize 256x2048}",
        ),
        to_connection("knn4", "ec4"),
        to_Conv(
            "cat",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(ec4-east)",
            width=2.0,
            height=22,
            depth=22,
            caption=r"Concat\\{\scriptsize 512x2048}",
        ),
        to_connection("ec4", "cat"),
        to_Conv(
            "mlp",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(cat-east)",
            width=1.9,
            height=22,
            depth=22,
            caption=r"MLP\\{\scriptsize 128x2048}",
        ),
        to_connection("cat", "mlp"),
        to_Conv(
            "head",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(mlp-east)",
            width=1.8,
            height=24,
            depth=24,
            caption=r"Class\\{\scriptsize 11x2048}",
        ),
        to_connection("mlp", "head"),
        to_end(),
    ]
    return arch, to_generate


def main():
    parser = argparse.ArgumentParser(description="Generate PlotNeuralNet TeX for DGCNN segmentation")
    parser.add_argument(
        "--plotnn_root",
        type=str,
        default="tools/PlotNeuralNet",
        help="Path to PlotNeuralNet root (contains pycore/ and layers/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagrams/plotneuralnet/dgcnn_segmentation.tex",
        help="Output .tex path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    plotnn_root = (root / args.plotnn_root).resolve()
    if not plotnn_root.exists():
        raise FileNotFoundError(f"PlotNeuralNet not found: {plotnn_root}")

    sys.path.insert(0, str(plotnn_root))

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rel_plotnn = Path(Path(os.path.relpath(plotnn_root, output_path.parent)).as_posix())

    arch, to_generate = build_arch(rel_plotnn)
    to_generate(arch, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
