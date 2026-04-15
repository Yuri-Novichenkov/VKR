"""
Generate an LDGCNN (segmentation) architecture diagram using PlotNeuralNet.

Output: TeX file that can be compiled with LaTeX/TikZ.
"""

import argparse
import os
import sys
from pathlib import Path


def build_arch(tools_path: Path, attention_label: str, k_small: int, k_large: int):
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
        to_Conv(
            "msec1",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(input-east)",
            width=1.8,
            height=32,
            depth=32,
            caption=rf"MSEC1\\{{\scriptsize 64x2048}}\\{{\scriptsize ks/kl={k_small}/{k_large}}}",
        ),
        to_connection("input", "msec1"),
        to_Conv(
            "msec2",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(msec1-east)",
            width=1.8,
            height=30,
            depth=30,
            caption=r"MSEC2\\{\scriptsize 64x2048}",
        ),
        to_connection("msec1", "msec2"),
    ]

    no_attn_labels = {"none", "noattn", "no_attn", "no-attn"}
    is_no_attn = attention_label.strip().lower() in no_attn_labels

    if not is_no_attn:
        arch.extend(
            [
                to_Pool(
                    "attn",
                    offset="(2.8,0,0)",
                    to="(msec2-east)",
                    width=1.2,
                    height=26,
                    depth=26,
                    caption=rf"{attention_label}\\{{\scriptsize 64x2048}}",
                ),
                to_connection("msec2", "attn"),
                to_Conv(
                    "msec3",
                    s_filer="",
                    n_filer="",
                    offset="(2.8,0,0)",
                    to="(attn-east)",
                    width=1.8,
                    height=26,
                    depth=26,
                    caption=r"MSEC3\\{\scriptsize 128x2048}",
                ),
                to_connection("attn", "msec3"),
            ]
        )
    else:
        arch.extend(
            [
                to_Conv(
                    "msec3",
                    s_filer="",
                    n_filer="",
                    offset="(2.8,0,0)",
                    to="(msec2-east)",
                    width=1.8,
                    height=26,
                    depth=26,
                    caption=r"MSEC3\\{\scriptsize 128x2048}",
                ),
                to_connection("msec2", "msec3"),
            ]
        )

    arch.extend(
        [
        to_Conv(
            "msec4",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(msec3-east)",
            width=1.8,
            height=24,
            depth=24,
            caption=r"MSEC4\\{\scriptsize 256x2048}",
        ),
        to_connection("msec3", "msec4"),
        to_Conv(
            "cat",
            s_filer="",
            n_filer="",
            offset="(2.8,0,0)",
            to="(msec4-east)",
            width=2.0,
            height=22,
            depth=22,
            caption=r"Concat\\{\scriptsize 512x2048}",
        ),
        to_connection("msec4", "cat"),
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
    ])
    return arch, to_generate


def main():
    parser = argparse.ArgumentParser(description="Generate PlotNeuralNet TeX for LDGCNN segmentation")
    parser.add_argument(
        "--plotnn_root",
        type=str,
        default="tools/PlotNeuralNet",
        help="Path to PlotNeuralNet root (contains pycore/ and layers/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagrams/plotneuralnet/ldgcnn_segmentation.tex",
        help="Output .tex path",
    )
    parser.add_argument(
        "--attention_label",
        type=str,
        default="Attn",
        help="Caption for attention block (e.g. Attn, GATv2, LocalWin, None)",
    )
    parser.add_argument("--k_small", type=int, default=20, help="k_small for multi-scale neighborhood")
    parser.add_argument("--k_large", type=int, default=40, help="k_large for multi-scale neighborhood")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    plotnn_root = (root / args.plotnn_root).resolve()
    if not plotnn_root.exists():
        raise FileNotFoundError(f"PlotNeuralNet not found: {plotnn_root}")

    sys.path.insert(0, str(plotnn_root))

    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rel_plotnn = Path(Path(os.path.relpath(plotnn_root, output_path.parent)).as_posix())

    arch, to_generate = build_arch(rel_plotnn, args.attention_label, args.k_small, args.k_large)
    to_generate(arch, str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
