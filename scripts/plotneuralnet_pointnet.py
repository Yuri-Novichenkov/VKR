"""
Generate a PointNet (segmentation) architecture diagram using PlotNeuralNet.

Output: TeX file that can be compiled with LaTeX/TikZ.
"""

import argparse
import os
import sys
from pathlib import Path


def build_arch(tools_path: Path):
    # PlotNeuralNet is imported dynamically after sys.path setup.
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
            height=38,
            depth=38,
            caption=r"Input\\{\scriptsize 9x4096}",
        ),
        to_Conv(
            "tnet_in",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(input-east)",
            width=1.8,
            height=34,
            depth=34,
            caption=r"TNet-in\\{\scriptsize 3x3}",
        ),
        to_connection("input", "tnet_in"),
        to_Conv(
            "mlp_64",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(tnet_in-east)",
            width=1.8,
            height=30,
            depth=30,
            caption=r"M64\\{\scriptsize 64x4096}",
        ),
        to_connection("tnet_in", "mlp_64"),
        to_Conv(
            "tnet_feat",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(mlp_64-east)",
            width=1.8,
            height=26,
            depth=26,
            caption=r"TNet-feat\\{\scriptsize 64x64}",
        ),
        to_connection("mlp_64", "tnet_feat"),
        to_Conv(
            "mlp_1024",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(tnet_feat-east)",
            width=2.0,
            height=22,
            depth=22,
            caption=r"M1024\\{\scriptsize 1024x4096}",
        ),
        to_connection("tnet_feat", "mlp_1024"),
        to_Pool(
            "global_max",
            offset="(2.8,0,0)",
            to="(mlp_1024-east)",
            width=1.2,
            height=18,
            depth=18,
            caption=r"GMax\\{\scriptsize 1024x1}",
        ),
        to_connection("mlp_1024", "global_max"),
        to_Conv(
            "fusion",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(global_max-east)",
            width=2.0,
            height=20,
            depth=20,
            caption=r"Fuse\\{\scriptsize 1088x4096}",
        ),
        to_connection("global_max", "fusion"),
        to_Conv(
            "head",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(fusion-east)",
            width=1.8,
            height=24,
            depth=24,
            caption=r"Class\\{\scriptsize 11x4096}",
        ),
        to_connection("fusion", "head"),
        to_end(),
    ]
    return arch, to_generate


def main():
    parser = argparse.ArgumentParser(description="Generate PlotNeuralNet TeX for PointNet segmentation")
    parser.add_argument(
        "--plotnn_root",
        type=str,
        default="tools/PlotNeuralNet",
        help="Path to PlotNeuralNet root (contains pycore/ and layers/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagrams/plotneuralnet/pointnet_segmentation.tex",
        help="Output .tex path",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    plotnn_root = (root / args.plotnn_root).resolve()
    if not plotnn_root.exists():
        raise FileNotFoundError(f"PlotNeuralNet not found: {plotnn_root}")

    # Make PlotNeuralNet importable: import pycore.tikzeng
    sys.path.insert(0, str(plotnn_root))

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a relative path for \subimport{.../layers/}{init}
    # to avoid encoding issues with absolute paths containing non-ASCII chars.
    rel_plotnn = Path(Path(os.path.relpath(plotnn_root, output_path.parent)).as_posix())

    arch, to_generate = build_arch(rel_plotnn)
    to_generate(arch, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
