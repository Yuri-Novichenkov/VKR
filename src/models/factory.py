"""
Фабрика моделей point-cloud.

Единая функция build_model используется во всех скриптах
(train.py, test.py, predictions.py), чтобы исключить расхождение
при независимом копировании кода.
"""

from .pointnet import PointNetSegmentation, PointNetClassification
from .pointnet_plusplus import PointNetPlusPlusSegmentation, PointNetPlusPlusClassification
from .dgcnn import DGCNNSegmentation, DGCNNClassification
from .ldgcnn import LDGCNNSegmentation, LDGCNNClassification
from .point_transformer import PointTransformerSegmentation, PointTransformerClassification
from .ldgcnn_flash import LDGCNNFlashSegmentation, LDGCNNFlashClassification


def build_model(
    model_type: str,
    task: str,
    num_classes: int,
    num_features: int,
    k: int = 20,
    k_small: int = 20,
    k_large: int = 40,
    attention_type: str = "none",
    attention_k: int = 16,
    attention_heads: int = 4,
    attention_dropout: float = 0.1,
    pt_k: int = 16,
    pt_channels: tuple = (32, 64, 128, 256, 512),
    fast_knn: bool = False,
):
    """Создаёт модель по строковому имени архитектуры и задачи.

    Args:
        model_type:       "pointnet" | "pointnet++" | "dgcnn" | "ldgcnn" | "pointtransformer" | "ldgcnn_flash"
        task:             "segmentation" | "classification"
        num_classes:      количество классов
        num_features:     количество входных каналов на точку
        k:                число ближайших соседей для DGCNN
        k_small/k_large:  масштабы kNN для LDGCNN
        attention_type:   тип attention-блока для LDGCNN
        attention_k:      число соседей в attention-окне
        attention_heads:  число голов attention
        attention_dropout: dropout в attention
        pt_k:             число соседей для Point Transformer (default 16)
        pt_channels:      размеры каналов по уровням для Point Transformer

    Returns:
        nn.Module — инициализированная модель.

    Raises:
        ValueError: при неизвестных model_type или task.
    """
    if task not in ("segmentation", "classification"):
        raise ValueError(f"Неизвестная задача: {task!r}. Ожидается 'segmentation' или 'classification'.")

    seg = task == "segmentation"

    if model_type == "pointnet":
        cls = PointNetSegmentation if seg else PointNetClassification
        return cls(num_classes=num_classes, num_features=num_features)

    if model_type == "pointnet++":
        cls = PointNetPlusPlusSegmentation if seg else PointNetPlusPlusClassification
        return cls(num_classes=num_classes, num_features=num_features)

    if model_type == "dgcnn":
        cls = DGCNNSegmentation if seg else DGCNNClassification
        return cls(num_classes=num_classes, num_features=num_features, k=k, fast_knn=fast_knn)

    if model_type == "ldgcnn":
        ldgcnn_kwargs = dict(
            num_classes=num_classes,
            num_features=num_features,
            k_small=k_small,
            k_large=k_large,
            attention_type=attention_type,
            attention_k=attention_k,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            fast_knn=fast_knn,
        )
        cls = LDGCNNSegmentation if seg else LDGCNNClassification
        return cls(**ldgcnn_kwargs)

    if model_type == "pointtransformer":
        pt_kwargs = dict(
            num_classes=num_classes,
            num_features=num_features,
            k=pt_k,
            channels=pt_channels,
            fast_knn=fast_knn,
        )
        cls = PointTransformerSegmentation if seg else PointTransformerClassification
        return cls(**pt_kwargs)

    if model_type == "ldgcnn_flash":
        cls = LDGCNNFlashSegmentation if seg else LDGCNNFlashClassification
        return cls(
            num_classes=num_classes,
            num_features=num_features,
            k=k,
            num_heads=attention_heads,
            dropout=attention_dropout,
            fast_knn=fast_knn,
        )

    raise ValueError(f"Неизвестная архитектура: {model_type!r}.")
