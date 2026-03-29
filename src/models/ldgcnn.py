"""
LDGCNN: расширение DGCNN с multi-scale соседствами (k_small/k_large).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.point_ops import get_graph_feature, EdgeConv, knn


def _gather_neighbors(x, idx):
    """
    Собирает признаки соседей по индексам.
    Args:
        x: (B, H, N, D)
        idx: (B, N, K)
    Returns:
        neighbors: (B, H, N, K, D)
    """
    bsz, heads, num_points, dim = x.shape
    k = idx.shape[-1]
    base = torch.arange(bsz, device=x.device, dtype=idx.dtype).view(bsz, 1, 1) * num_points
    flat_idx = (idx + base).reshape(-1)
    x_nodes = x.permute(0, 2, 1, 3).reshape(bsz * num_points, heads, dim)
    neighbors = x_nodes[flat_idx]
    neighbors = neighbors.reshape(bsz, num_points, k, heads, dim).permute(0, 3, 1, 2, 4).contiguous()
    return neighbors


class _GATv2LocalAttention(nn.Module):
    """
    Локальный GATv2-подобный attention на kNN-окрестности.
    """

    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) должен делиться на heads ({heads})")
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.attn_vec = nn.Parameter(torch.empty(heads, 2 * self.dim_head))
        nn.init.xavier_uniform_(self.attn_vec)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        # x: (B, C, N), idx: (B, N, K)
        bsz, _, num_points = x.shape
        k = idx.shape[-1]

        q = self.q_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()
        k_proj = self.k_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()
        v = self.v_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()

        q_i = q.unsqueeze(3).expand(-1, -1, -1, k, -1)  # (B, H, N, K, D)
        k_j = _gather_neighbors(k_proj, idx)
        v_j = _gather_neighbors(v, idx)

        logits_input = torch.cat([q_i, k_j], dim=-1)  # (B, H, N, K, 2D)
        logits = F.leaky_relu((logits_input * self.attn_vec.view(1, self.heads, 1, 1, -1)).sum(dim=-1), negative_slope=0.2)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.sum(attn.unsqueeze(-1) * v_j, dim=3)  # (B, H, N, D)
        out = out.permute(0, 1, 3, 2).reshape(bsz, self.channels, num_points)
        out = self.out_proj(out)
        return out


class _LocalWindowAttention(nn.Module):
    """
    Облегченный локальный multi-head scaled dot-product attention.
    """

    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        if channels % heads != 0:
            raise ValueError(f"channels ({channels}) должен делиться на heads ({heads})")
        self.channels = channels
        self.heads = heads
        self.dim_head = channels // heads
        self.scale = self.dim_head ** -0.5

        self.q_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, idx):
        # x: (B, C, N), idx: (B, N, K)
        bsz, _, num_points = x.shape
        k = idx.shape[-1]

        q = self.q_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()
        k_proj = self.k_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()
        v = self.v_proj(x).view(bsz, self.heads, self.dim_head, num_points).permute(0, 1, 3, 2).contiguous()

        q_i = q.unsqueeze(3).expand(-1, -1, -1, k, -1)
        k_j = _gather_neighbors(k_proj, idx)
        v_j = _gather_neighbors(v, idx)

        scores = torch.sum(q_i * k_j, dim=-1) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.sum(attn.unsqueeze(-1) * v_j, dim=3)  # (B, H, N, D)
        out = out.permute(0, 1, 3, 2).reshape(bsz, self.channels, num_points)
        out = self.out_proj(out)
        return out


class _MultiScaleEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_small=20, k_large=40):
        super().__init__()
        self.k_small = k_small
        self.k_large = k_large
        self.ec_small = EdgeConv(in_channels=in_channels, out_channels=out_channels)
        self.ec_large = EdgeConv(in_channels=in_channels, out_channels=out_channels)
        self.fuse = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        # x: (B, C, N)
        # Две окрестности разного масштаба помогают одновременно
        # захватывать локальные и более "широкие" контексты.
        x_small = self.ec_small(get_graph_feature(x, k=self.k_small))
        x_large = self.ec_large(get_graph_feature(x, k=self.k_large))
        x = torch.cat([x_small, x_large], dim=1)
        x = self.fuse(x)
        return x


class LDGCNNSegmentation(nn.Module):
    """
    LDGCNN для семантической сегментации (мульти-скейл EdgeConv).
    """
    def __init__(
        self,
        num_classes,
        num_features=9,
        k_small=20,
        k_large=40,
        dropout=0.5,
        attention_type="none",
        attention_k=16,
        attention_heads=4,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.attention_type = attention_type.lower()
        self.attention_k = attention_k

        self.ec1 = _MultiScaleEdgeConv(in_channels=2 * num_features, out_channels=64, k_small=k_small, k_large=k_large)
        self.ec2 = _MultiScaleEdgeConv(in_channels=2 * 64, out_channels=64, k_small=k_small, k_large=k_large)
        self.ec3 = _MultiScaleEdgeConv(in_channels=2 * 64, out_channels=128, k_small=k_small, k_large=k_large)
        self.ec4 = _MultiScaleEdgeConv(in_channels=2 * 128, out_channels=256, k_small=k_small, k_large=k_large)
        if self.attention_type == "gatv2":
            self.attention = _GATv2LocalAttention(64, heads=attention_heads, dropout=attention_dropout)
        elif self.attention_type == "local_window":
            self.attention = _LocalWindowAttention(64, heads=attention_heads, dropout=attention_dropout)
        else:
            self.attention = None
        self.attn_norm = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv1d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()  # (B, F, N)

        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        if self.attention is not None:
            # Внимание считается по локальной kNN-окрестности текущих признаков.
            k_local = min(self.attention_k, x2.shape[-1])
            attn_idx = knn(x2, k=k_local)
            x2 = self.attn_norm(x2 + self.attention(x2, attn_idx))
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x_cat)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        return x

    def get_loss(self, predictions, targets):
        B, N, num_classes = predictions.shape
        predictions_flat = predictions.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(predictions_flat, targets_flat)
        return loss, loss, torch.tensor(0.0, device=predictions.device)


class LDGCNNClassification(nn.Module):
    """
    LDGCNN для классификации облаков точек (мульти-скейл EdgeConv).
    """
    def __init__(
        self,
        num_classes,
        num_features=9,
        k_small=20,
        k_large=40,
        emb_dims=1024,
        dropout=0.5,
        attention_type="none",
        attention_k=16,
        attention_heads=4,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.attention_type = attention_type.lower()
        self.attention_k = attention_k

        self.ec1 = _MultiScaleEdgeConv(in_channels=2 * num_features, out_channels=64, k_small=k_small, k_large=k_large)
        self.ec2 = _MultiScaleEdgeConv(in_channels=2 * 64, out_channels=64, k_small=k_small, k_large=k_large)
        self.ec3 = _MultiScaleEdgeConv(in_channels=2 * 64, out_channels=128, k_small=k_small, k_large=k_large)
        self.ec4 = _MultiScaleEdgeConv(in_channels=2 * 128, out_channels=256, k_small=k_small, k_large=k_large)
        if self.attention_type == "gatv2":
            self.attention = _GATv2LocalAttention(64, heads=attention_heads, dropout=attention_dropout)
        elif self.attention_type == "local_window":
            self.attention = _LocalWindowAttention(64, heads=attention_heads, dropout=attention_dropout)
        else:
            self.attention = None
        self.attn_norm = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.fc1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()

        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        if self.attention is not None:
            k_local = min(self.attention_k, x2.shape[-1])
            attn_idx = knn(x2, k=k_local)
            x2 = self.attn_norm(x2 + self.attention(x2, attn_idx))
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x_cat)

        x_max = torch.max(x, dim=2)[0]
        x_avg = torch.mean(x, dim=2)
        x = torch.cat((x_max, x_avg), dim=1)

        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.fc3(x)
        return x

    def get_loss(self, predictions, targets):
        loss = F.cross_entropy(predictions, targets)
        return loss, loss, torch.tensor(0.0, device=predictions.device)
