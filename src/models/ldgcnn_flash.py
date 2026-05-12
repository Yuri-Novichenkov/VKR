"""
LDGCNNFlash: LDGCNN + Flash Self-Attention.

Кастомная модель для ВКР: сочетает локальную экстракцию признаков
из LDGCNN с глобальным контекстом через Flash Self-Attention.

── Идея ──────────────────────────────────────────────────────────────────────
LDGCNN (Linked Dynamic Graph CNN) строит динамический граф в XYZ-пространстве
и связывает (links) признаки разных уровней в качестве входа для следующего.
Это даёт богатый локальный дескриптор, но ограничено локальной окрестностью.

Flash Self-Attention добавляет глобальный контекст: каждая точка может
«смотреть» на все N=4096 точек одновременно, без создания матрицы N×N.

── Архитектура (сегментация, 3 стадии) ──────────────────────────────────────
    Вход x: (B, N, F)

    Stage 1:  EdgeConv(F → C1, k_edge)  →  FlashAttnBlock(C1)  →  a1 (B,N,C1)
    Stage 2:  EdgeConv(F+C1 → C2, k)   →  FlashAttnBlock(C2)  →  a2 (B,N,C2)
    Stage 3:  EdgeConv(F+C1+C2 → C3, k)→  FlashAttnBlock(C3)  →  a3 (B,N,C3)

    Head:  cat([a1, a2, a3]) → Linear(C1+C2+C3, 256) → BN → ReLU → Linear(256, num_classes)

Linked Features (как в LDGCNN):
    Вход каждого EdgeConv = конкатенация исходных признаков + выходов
    всех предыдущих FlashAttn блоков.

── Flash Attention ───────────────────────────────────────────────────────────
    F.scaled_dot_product_attention (PyTorch ≥ 2.0) автоматически использует
    Flash Attention kernel на CUDA при dtype=float16:
      - O(N·d) памяти (tile-based, без материализации матрицы N×N)
      - ~3-4× быстрее наивного attention при N=4096
      - AMP-совместимо (FP16 входы поддерживаются)

── Численная стабильность под AMP ────────────────────────────────────────────
    - LayerNorm вычисляется явно в FP32 (как softmax в Point Transformer)
    - BatchNorm в EdgeConv стабилен: видит B*N*k > 100k сэмплов
    - _knn_query использует фиксированный _square_distance (autocast=False)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Импортируем исправленные утилиты из point_transformer:
# _square_distance уже обёрнут в autocast(enabled=False) для корректного FP32.
from .point_transformer import _knn_query, _index_points, _apply_bn


# ── EdgeConv ─────────────────────────────────────────────────────────────────

class _EdgeConvMLP(nn.Module):
    """
    Shared MLP для обработки edge-признаков: Linear → BN → LeakyReLU.
    Работает с плоским (B*N*k, in_ch) тензором — BN всегда получает
    достаточно сэмплов (при N=4096, k=20 → 80k+ сэмплов на батч).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, out_ch, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EdgeConvBlock(nn.Module):
    """
    LDGCNN-стиль EdgeConv: граф строится в XYZ-пространстве (не в feature space).

    Для каждой точки i с k соседями j:
        edge_{ij} = [x_i ‖ x_j − x_i]          ← относительный дескриптор
        out_i = max_j MLP(edge_{ij})             ← агрегация max-pool

    Параметры:
        in_channels:  размерность входных признаков x
        out_channels: размерность выходных признаков
        k:            число ближайших соседей
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20, fast_knn: bool = False):
        super().__init__()
        self.k        = k
        self.fast_knn = fast_knn
        # 2*in_channels: [x_i ‖ x_j − x_i]
        self.mlp = _EdgeConvMLP(2 * in_channels, out_channels)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:   (B, N, C_in)  — входные признаки
            pos: (B, N, 3)     — XYZ-координаты для построения графа
        Returns:
            (B, N, C_out)
        """
        B, N, C = x.shape
        K = self.k

        # k-NN в XYZ-пространстве (исправленный _knn_query — FP32)
        idx = _knn_query(K, pos, pos, fast=self.fast_knn)  # (B, N, K)
        idx_flat = idx.reshape(B, N * K)

        x_j = _index_points(x, idx_flat).view(B, N, K, C)  # (B, N, K, C)
        x_i = x.unsqueeze(2).expand_as(x_j)                # (B, N, K, C)

        # Edge features: [x_i ‖ x_j − x_i]
        edge = torch.cat([x_i, x_j - x_i], dim=-1)         # (B, N, K, 2C)

        # Shared MLP + max-pool
        out = self.mlp(edge.reshape(B * N * K, -1))          # (B*N*K, C_out)
        out = out.reshape(B, N, K, -1).max(dim=2)[0]         # (B, N, C_out)
        return out


# ── Flash Self-Attention Block ────────────────────────────────────────────────

class FlashAttentionBlock(nn.Module):
    """
    Multi-head Self-Attention с Flash Attention backend + FFN.

    Использует F.scaled_dot_product_attention (PyTorch ≥ 2.0), который
    на CUDA с FP16 автоматически выбирает Flash Attention kernel:
      - Tile-based вычисление: O(N·d) памяти вместо O(N²)
      - Не материализует матрицу внимания N×N в HBM
      - Поддерживает head_dim от 8 до 128

    Структура блока (Pre-LN Transformer):
        x → LayerNorm → MHSA → Dropout → + x    (Self-Attention с residual)
        x → LayerNorm → FFN  → Dropout → + x    (Feed-Forward с residual)

    ВАЖНО: LayerNorm вызывается напрямую без принудительного FP32-каста.
    Под AMP он автоматически остаётся в FP32 (fp32-preserved op в autocast).
    При model.half() inference работает в FP16 на CUDA без dtype-mismatch.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) должен делиться на num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout_p = dropout

        # ── Self-Attention ────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * channels, bias=False)
        self.attn_proj = nn.Linear(channels, channels, bias=False)

        # ── Feed-Forward Network ──────────────────────────────────────────
        self.norm2 = nn.LayerNorm(channels)
        ffn_dim = channels * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, channels, bias=False),
        )

        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape

        # ── Self-Attention ────────────────────────────────────────────────
        # LayerNorm: под AMP автоматически остаётся в FP32 (fp32-preserved op).
        # При model.half() принимает FP16 — работает корректно на CUDA.
        # Явный .float() здесь вызывал dtype-mismatch при model.half() inference.
        x_norm = self.norm1(x)

        # QKV проекция
        qkv = self.qkv(x_norm)                                   # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                         # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)                                   # (B, H, N, D)

        # Flash Attention: автоматически на CUDA + FP16
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )                                                          # (B, H, N, D)

        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)      # (B, N, C)
        attn_out = self.drop(self.attn_proj(attn_out))
        x = x + attn_out                                           # residual

        # ── Feed-Forward ──────────────────────────────────────────────────
        x_norm2 = self.norm2(x)
        ffn_out = self.drop(self.ffn(x_norm2))
        x = x + ffn_out                                            # residual

        return x


# ── Segmentation Model ────────────────────────────────────────────────────────

class LDGCNNFlashSegmentation(nn.Module):
    """
    LDGCNNFlash для семантической сегментации.

    3 стадии: EdgeConv (локальные признаки) + Flash Self-Attention (глобальный контекст).
    Linked Features: каждый EdgeConv получает исходные признаки + выходы всех предыдущих стадий.

    Параметры:
        num_classes:  количество классов
        num_features: размерность входных признаков на точку (default 9)
        k:            число ближайших соседей для EdgeConv (default 20)
        channels:     размеры каналов по стадиям (default (64, 128, 256))
        num_heads:    число голов Flash Attention (default 4)
        ffn_ratio:    множитель скрытого слоя FFN (default 2)
        dropout:      dropout в attention и FFN (default 0.1)
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int = 9,
        k: int = 20,
        channels: tuple = (64, 128, 256),
        num_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        fast_knn: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        C1, C2, C3 = channels

        # ── Stage 1 ───────────────────────────────────────────────────────
        self.ec1 = EdgeConvBlock(num_features, C1, k=k, fast_knn=fast_knn)
        self.attn1 = FlashAttentionBlock(C1, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        # ── Stage 2 (linked: [x, a1]) ────────────────────────────────────
        self.ec2 = EdgeConvBlock(num_features + C1, C2, k=k, fast_knn=fast_knn)
        self.attn2 = FlashAttentionBlock(C2, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        # ── Stage 3 (linked: [x, a1, a2]) ────────────────────────────────
        self.ec3 = EdgeConvBlock(num_features + C1 + C2, C3, k=k, fast_knn=fast_knn)
        self.attn3 = FlashAttentionBlock(C3, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        # ── Head MLP ─────────────────────────────────────────────────────
        agg_ch = C1 + C2 + C3
        self.head = nn.Sequential(
            nn.Linear(agg_ch, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, num_features)
        Returns:
            (B, N, num_classes) — логиты
        """
        B, N, _ = x.shape
        pos = x[:, :, :3].contiguous()   # XYZ для построения графа

        # Stage 1
        a1 = self.attn1(self.ec1(x, pos))                           # (B, N, C1)

        # Stage 2: linked
        a2 = self.attn2(self.ec2(torch.cat([x, a1], dim=-1), pos))  # (B, N, C2)

        # Stage 3: linked
        a3 = self.attn3(self.ec3(torch.cat([x, a1, a2], dim=-1), pos))  # (B, N, C3)

        # Head
        agg = torch.cat([a1, a2, a3], dim=-1)                        # (B, N, C1+C2+C3)
        return self.head(agg.reshape(B * N, -1)).reshape(B, N, -1)   # (B, N, num_classes)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_weights=None,
    ) -> tuple:
        B, N, C = predictions.shape
        loss = F.cross_entropy(
            predictions.reshape(-1, C),
            targets.reshape(-1),
            weight=class_weights,
        )
        return loss, loss, torch.tensor(0.0, device=predictions.device)


# ── Classification Model ──────────────────────────────────────────────────────

class LDGCNNFlashClassification(nn.Module):
    """
    LDGCNNFlash для классификации облаков точек.
    Тот же encoder, что у сегментации, затем global max+avg pooling → MLP.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int = 9,
        k: int = 20,
        channels: tuple = (64, 128, 256),
        num_heads: int = 4,
        ffn_ratio: int = 2,
        dropout: float = 0.1,
        fast_knn: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        C1, C2, C3 = channels

        self.ec1 = EdgeConvBlock(num_features, C1, k=k, fast_knn=fast_knn)
        self.attn1 = FlashAttentionBlock(C1, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        self.ec2 = EdgeConvBlock(num_features + C1, C2, k=k, fast_knn=fast_knn)
        self.attn2 = FlashAttentionBlock(C2, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        self.ec3 = EdgeConvBlock(num_features + C1 + C2, C3, k=k, fast_knn=fast_knn)
        self.attn3 = FlashAttentionBlock(C3, num_heads=num_heads, ffn_ratio=ffn_ratio, dropout=dropout)

        agg_ch = (C1 + C2 + C3) * 2  # max + avg pooling
        self.head = nn.Sequential(
            nn.Linear(agg_ch, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        pos = x[:, :, :3].contiguous()

        a1 = self.attn1(self.ec1(x, pos))
        a2 = self.attn2(self.ec2(torch.cat([x, a1], dim=-1), pos))
        a3 = self.attn3(self.ec3(torch.cat([x, a1, a2], dim=-1), pos))

        agg = torch.cat([a1, a2, a3], dim=-1)          # (B, N, C1+C2+C3)
        global_feat = torch.cat([
            agg.max(dim=1)[0],
            agg.mean(dim=1),
        ], dim=-1)                                       # (B, 2*(C1+C2+C3))
        return self.head(global_feat)

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_weights=None,
    ) -> tuple:
        loss = F.cross_entropy(predictions, targets, weight=class_weights)
        return loss, loss, torch.tensor(0.0, device=predictions.device)
