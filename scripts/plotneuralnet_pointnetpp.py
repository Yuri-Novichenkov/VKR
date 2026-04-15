"""
Generate a PointNet++ (segmentation) architecture diagram using PlotNeuralNet.

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
            height=38,
            depth=38,
            caption=r"Input\\{\scriptsize 9x4096}",
        ),
        to_Conv(
            "sa1",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(input-east)",
            width=1.9,
            height=34,
            depth=34,
            caption=r"SA1\\{\scriptsize 128x512}",
        ),
        to_connection("input", "sa1"),
        to_Conv(
            "sa2",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(sa1-east)",
            width=1.9,
            height=30,
            depth=30,
            caption=r"SA2\\{\scriptsize 256x256}",
        ),
        to_connection("sa1", "sa2"),
        to_Pool(
            "sa3_global",
            offset="(2.8,0,0)",
            to="(sa2-east)",
            width=1.2,
            height=24,
            depth=24,
            caption=r"SA3\\{\scriptsize 1024x1}",
        ),
        to_connection("sa2", "sa3_global"),
        to_Conv(
            "fp3",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(sa3_global-east)",
            width=1.9,
            height=28,
            depth=28,
            caption=r"FP3\\{\scriptsize 256x256}",
        ),
        to_connection("sa3_global", "fp3"),
        to_Conv(
            "fp2",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(fp3-east)",
            width=1.9,
            height=32,
            depth=32,
            caption=r"FP2\\{\scriptsize 128x512}",
        ),
        to_connection("fp3", "fp2"),
        to_Conv(
            "fp1",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(fp2-east)",
            width=1.9,
            height=36,
            depth=36,
            caption=r"FP1\\{\scriptsize 128x4096}",
        ),
        to_connection("fp2", "fp1"),
        to_Conv(
            "head",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(fp1-east)",
            width=1.8,
            height=26,
            depth=26,
            caption=r"Class\\{\scriptsize 11x4096}",
        ),
        to_connection("fp1", "head"),
        to_end(),
    ]
    return arch, to_generate


def main():
    parser = argparse.ArgumentParser(description="Generate PlotNeuralNet TeX for PointNet++ segmentation")
    parser.add_argument(
        "--plotnn_root",
        type=str,
        default="tools/PlotNeuralNet",
        help="Path to PlotNeuralNet root (contains pycore/ and layers/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagrams/plotneuralnet/pointnetpp_segmentation.tex",
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
