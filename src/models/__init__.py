"""Экспорт моделей сегментации и классификации, используемых в скриптах."""

from .pointnet import PointNetSegmentation, PointNetClassification, TNet
from .pointnet_plusplus import PointNetPlusPlusSegmentation, PointNetPlusPlusClassification
from .dgcnn import DGCNNSegmentation, DGCNNClassification
from .ldgcnn import LDGCNNSegmentation, LDGCNNClassification

__all__ = [
    "PointNetSegmentation",
    "PointNetClassification",
    "PointNetPlusPlusSegmentation",
    "PointNetPlusPlusClassification",
    "DGCNNSegmentation",
    "DGCNNClassification",
    "LDGCNNSegmentation",
    "LDGCNNClassification",
    "TNet",
]
