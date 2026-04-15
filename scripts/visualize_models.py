"""
Генерация схематичных графов моделей через torchviz и hiddenlayer.

Скрипт создает визуализации для PointNet / PointNet++ / DGCNN / LDGCNN
в задачах segmentation и classification.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import (  # noqa: E402
    DGCNNClassification,
    DGCNNSegmentation,
    LDGCNNClassification,
    LDGCNNSegmentation,
    PointNetClassification,
    PointNetPlusPlusClassification,
    PointNetPlusPlusSegmentation,
    PointNetSegmentation,
)


class _ForwardAdapter(nn.Module):
    """Приводит выход модели к одному тензору для визуализаторов."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise TypeError("Ожидался tensor-выход модели для построения графа.")
        return out


def _build_model(name: str, task: str, args) -> nn.Module:
    if task == "segmentation":
        if name == "pointnet":
            return PointNetSegmentation(num_classes=args.num_classes, num_features=args.num_features)
        if name == "pointnet++":
            return PointNetPlusPlusSegmentation(num_classes=args.num_classes, num_features=args.num_features)
        if name == "dgcnn":
            return DGCNNSegmentation(num_classes=args.num_classes, num_features=args.num_features, k=args.k)
        if name == "ldgcnn":
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
    elif task == "classification":
        if name == "pointnet":
            return PointNetClassification(num_classes=args.num_classes, num_features=args.num_features)
        if name == "pointnet++":
            return PointNetPlusPlusClassification(num_classes=args.num_classes, num_features=args.num_features)
        if name == "dgcnn":
            return DGCNNClassification(num_classes=args.num_classes, num_features=args.num_features, k=args.k)
        if name == "ldgcnn":
            return LDGCNNClassification(
                num_classes=args.num_classes,
                num_features=args.num_features,
                k_small=args.k_small,
                k_large=args.k_large,
                attention_type=args.attention_type,
                attention_k=args.attention_k,
                attention_heads=args.attention_heads,
                attention_dropout=args.attention_dropout,
            )
    raise ValueError(f"Неизвестная комбинация model={name}, task={task}")


def _build_graph_name(model_name: str, task: str, args) -> str:
    base = f"{model_name.replace('+', 'p')}_{task}"
    if model_name == "dgcnn":
        return f"{base}_k{args.k}"
    if model_name == "ldgcnn":
        drop = str(args.attention_dropout).replace(".", "p")
        return (
            f"{base}_ks{args.k_small}_kl{args.k_large}"
            f"_attn-{args.attention_type}"
            f"_ak{args.attention_k}_ah{args.attention_heads}_ad{drop}"
        )
    return base


def _save_torchviz(model: nn.Module, sample: torch.Tensor, out_dir: Path, graph_name: str, fmt: str) -> None:
    try:
        from torchviz import make_dot
    except Exception as exc:
        raise RuntimeError("torchviz не установлен. Установите: pip install torchviz") from exc

    wrapped_model = _ForwardAdapter(model).eval()
    output = wrapped_model(sample)
    dot = make_dot(output.sum(), params=dict(wrapped_model.named_parameters()))

    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / f"{graph_name}.dot"
    dot.save(str(dot_path))

    # Рендер в png/svg требует системный Graphviz (утилиту dot).
    # Если dot не найден, оставляем .dot файл для импорта в draw.io/Graphviz Online.
    try:
        dot.render(filename=graph_name, directory=str(out_dir), format=fmt, cleanup=True)
    except Exception as exc:
        print(f"  WARN torchviz render skipped ({exc}). DOT сохранен: {dot_path}")


def _save_hiddenlayer(model: nn.Module, sample: torch.Tensor, out_dir: Path, graph_name: str) -> None:
    try:
        import hiddenlayer as hl
    except ImportError as exc:
        raise RuntimeError("hiddenlayer не установлен. Установите: pip install hiddenlayer") from exc

    # Совместимость hiddenlayer 0.3 с новыми версиями PyTorch.
    if not hasattr(torch.onnx, "_optimize_trace"):
        torch.onnx._optimize_trace = lambda trace, *_args, **_kwargs: trace

    # В новых версиях PyTorch torch._C.Node больше не поддерживает node["attr"].
    # Подменяем importer hiddenlayer на совместимую версию без чтения params.
    import hiddenlayer.pytorch_builder as hb

    def _import_graph_compat(hl_graph, model_obj, args_obj, input_names=None, verbose=False):
        del input_names, verbose  # Для совместимости сигнатуры.
        trace, _out = torch.jit._get_trace_graph(model_obj, args_obj)
        torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        if hasattr(torch_graph, "graph"):
            torch_graph = torch_graph.graph()

        nodes_list = list(torch_graph.nodes())
        for torch_node in nodes_list:
            op = torch_node.kind()
            outputs = [o.unique() for o in torch_node.outputs()]
            shape = hb.get_shape(torch_node)
            hl_node = hb.Node(
                uid=hb.pytorch_id(torch_node),
                name=None,
                op=op,
                output_shape=shape,
                params={},
            )
            hl_graph.add_node(hl_node)
            for target_torch_node in nodes_list:
                target_inputs = [i.unique() for i in target_torch_node.inputs()]
                if set(outputs) & set(target_inputs):
                    hl_graph.add_edge_by_id(
                        hb.pytorch_id(torch_node),
                        hb.pytorch_id(target_torch_node),
                        shape,
                    )
        return hl_graph

    hb.import_graph = _import_graph_compat

    wrapped_model = _ForwardAdapter(model).eval()
    graph = hl.build_graph(wrapped_model, sample)
    out_dir.mkdir(parents=True, exist_ok=True)
    dot = graph.build_dot()
    dot_path = out_dir / f"{graph_name}.dot"
    dot.save(str(dot_path))
    try:
        dot.render(filename=graph_name, directory=str(out_dir), format="png", cleanup=True)
    except Exception as exc:
        print(f"  WARN hiddenlayer render skipped ({exc}). DOT сохранен: {dot_path}")


def main():
    parser = argparse.ArgumentParser(description="Визуализация архитектур моделей через torchviz и hiddenlayer")
    parser.add_argument("--models", type=str, default="all", help="all или список через запятую: pointnet,pointnet++,dgcnn,ldgcnn")
    parser.add_argument("--task", type=str, default="segmentation", choices=["segmentation", "classification", "both"])
    parser.add_argument("--num_points", type=int, default=1024, help="Число точек N во входе (B, N, F)")
    parser.add_argument("--num_features", type=int, default=9, help="Число признаков F во входе (B, N, F)")
    parser.add_argument("--num_classes", type=int, default=8, help="Число классов для головы модели")
    parser.add_argument("--batch_size", type=int, default=1, help="Размер батча для dummy-входа")
    parser.add_argument("--output_dir", type=str, default="diagrams", help="Папка для сохранения схем")
    parser.add_argument("--torchviz_format", type=str, default="png", choices=["png", "svg"])
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["both", "torchviz", "hiddenlayer"],
        help="Какой визуализатор использовать",
    )

    parser.add_argument("--k", type=int, default=20, help="k для DGCNN")
    parser.add_argument("--k_small", type=int, default=20, help="k_small для LDGCNN")
    parser.add_argument("--k_large", type=int, default=40, help="k_large для LDGCNN")
    parser.add_argument("--attention_type", type=str, default="none", choices=["none", "gatv2", "local_window"])
    parser.add_argument("--attention_k", type=int, default=16)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    args = parser.parse_args()

    all_models = ["pointnet", "pointnet++", "dgcnn", "ldgcnn"]
    selected_models = all_models if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    for model_name in selected_models:
        if model_name not in all_models:
            raise ValueError(f"Неизвестная модель в --models: {model_name}")

    tasks = ["segmentation", "classification"] if args.task == "both" else [args.task]
    sample = torch.randn(args.batch_size, args.num_points, args.num_features)
    root = Path(args.output_dir)

    for task in tasks:
        for model_name in selected_models:
            model = _build_model(model_name, task, args).cpu().eval()
            graph_name = _build_graph_name(model_name, task, args)

            if args.backend in ("both", "torchviz"):
                print(f"[torchviz] {graph_name}")
                try:
                    _save_torchviz(
                        model=model,
                        sample=sample,
                        out_dir=root / "torchviz" / task,
                        graph_name=graph_name,
                        fmt=args.torchviz_format,
                    )
                    print(f"  OK: файлы в {root / 'torchviz' / task}")
                except Exception as exc:
                    print(f"  ERROR torchviz: {exc}")

            if args.backend in ("both", "hiddenlayer"):
                print(f"[hiddenlayer] {graph_name}")
                try:
                    _save_hiddenlayer(
                        model=model,
                        sample=sample,
                        out_dir=root / "hiddenlayer" / task,
                        graph_name=graph_name,
                    )
                    print(f"  OK: файлы в {root / 'hiddenlayer' / task}")
                except Exception as exc:
                    print(f"  ERROR hiddenlayer: {exc}")


if __name__ == "__main__":
    main()
