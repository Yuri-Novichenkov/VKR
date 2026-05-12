"""
Point Transformer для семантической сегментации 3D облаков точек.
Zhao et al. "Point Transformer" (ICCV 2021). arxiv.org/abs/2012.09164

Ключевая идея: vector self-attention с относительным позиционным кодированием.
Формула: y_i = Σ_j softmax(γ(φ(x_i) - ψ(x_j) + δ(p_i−p_j))) ⊙ (α(x_j) + δ(p_i−p_j))

Архитектура (сегментация): U-Net
  Энкодер — 5 стадий: 4096→1024→256→64→16 точек, каналы [32,64,128,256,512]
  Декодер — 4 стадии: интерполяция + skip-connection + PT Block

Примечания по численной стабильности:
  - softmax явно вычисляется в FP32 (FP16 переполняется при значениях > ~10)
  - используется BatchNorm1d (вместо LayerNorm) как в оригинальной статье,
    более стабилен с AMP
  - BN внутри MLPs delta и gamma предотвращает рост магнитуд активаций
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── вспомогательные операции над точками ────────────────────────────────────

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Квадраты попарных расстояний, ВСЕГДА в FP32.

    Args:
        src: (B, N, C)
        dst: (B, M, C)
    Returns:
        (B, N, M) — float32

    ВАЖНО: torch.matmul под AMP принудительно работает в FP16 даже при FP32-входах
    (matmul входит в список autocast-eligible ops). При малых расстояниях (< ~6e-5)
    FP16-результат underflow'ится в 0, что при IDW даёт 1/0 = inf → NaN.
    Явно отключаем autocast для этой операции.
    """
    src_f = src.float()
    dst_f = dst.float()
    device_type = src.device.type  # 'cuda' или 'cpu'
    with torch.amp.autocast(device_type=device_type, enabled=False):
        dist = -2.0 * torch.matmul(src_f, dst_f.permute(0, 2, 1))
        dist += torch.sum(src_f ** 2, dim=-1, keepdim=True)
        dist += torch.sum(dst_f ** 2, dim=-1, keepdim=True).permute(0, 2, 1)
    return dist.clamp(min=0.0)


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Индексирование набора точек.
    Args:
        points: (B, N, C)
        idx:    (B, M) или (B, M, K)
    Returns:
        (B, M, C) или (B, M, K, C)
    """
    B = points.shape[0]
    device = points.device
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_idx = (
        torch.arange(B, dtype=torch.long, device=device)
        .view(*view_shape)
        .expand_as(idx)
    )
    return points[batch_idx, idx]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS).
    Args:
        xyz:    (B, N, 3)
        npoint: число выбираемых точек
    Returns:
        centroids: (B, npoint) — индексы
    """
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # расстояния считаем в FP32 для стабильности
    xyz_f32 = xyz.float()
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz_f32[batch_idx, farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((xyz_f32 - centroid) ** 2, dim=-1)   # (B, N)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def _knn_query(k: int, pos_base: torch.Tensor, pos_query: torch.Tensor,
               fast: bool = False) -> torch.Tensor:
    """
    k ближайших соседей из pos_base для каждой точки pos_query.
    Args:
        k:          число соседей
        pos_base:   (B, N, 3) — база поиска
        pos_query:  (B, M, 3) — запросы
        fast:       использовать torch_cluster вместо полной матрицы расстояний
    Returns:
        idx: (B, M, k)
    """
    if fast:
        return _knn_query_tc(k, pos_base, pos_query)
    dist = _square_distance(pos_query.float(), pos_base.float())  # FP32 для стабильности
    _, idx = dist.topk(k, dim=-1, largest=False, sorted=True)
    return idx


def _knn_query_tc(k: int, pos_base: torch.Tensor, pos_query: torch.Tensor) -> torch.Tensor:
    """
    Быстрый kNN через torch_cluster. pos: (B, N/M, 3) -> idx: (B, M, k)
    """
    from torch_cluster import knn as tc_knn
    B, N, C = pos_base.shape
    M = pos_query.shape[1]
    device = pos_base.device
    base_flat  = pos_base.float().reshape(B * N, C)
    query_flat = pos_query.float().reshape(B * M, C)
    batch_base  = torch.arange(B, device=device).repeat_interleave(N)
    batch_query = torch.arange(B, device=device).repeat_interleave(M)
    # knn(x=база, y=запросы, k): для каждой точки y находит k ближайших в x
    edge_index = tc_knn(base_flat, query_flat, k, batch_x=batch_base, batch_y=batch_query)
    batch_offset = batch_query[edge_index[0]] * N
    local_neighbors = (edge_index[1] - batch_offset).view(B * M, k)
    return local_neighbors.view(B, M, k)


def _apply_bn(x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
    """
    Применяет BatchNorm1d к тензору произвольной формы (..., C).
    Решает проблему несовместимости BN1d с многомерными тензорами.
    """
    shape = x.shape
    return bn(x.reshape(-1, shape[-1])).reshape(shape)


# ── внутренний MLP с BN ──────────────────────────────────────────────────────

class _MLP(nn.Module):
    """
    2-слойный MLP: Linear → BN → ReLU → Linear.
    Работает с тензорами произвольной формы (..., in_ch).
    BN предотвращает взрывной рост активаций при обучении с AMP.
    """

    def __init__(self, in_ch: int, hid_ch: int, out_ch: int):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hid_ch, bias=False)
        self.bn1 = nn.BatchNorm1d(hid_ch)
        self.fc2 = nn.Linear(hid_ch, out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        # reshape в 2D для BN, затем восстановить форму
        out = F.relu(_apply_bn(self.fc1(x.reshape(-1, shape[-1])), self.bn1), inplace=True)
        return self.fc2(out).reshape(*shape[:-1], -1)


# ── Point Transformer Layer ──────────────────────────────────────────────────

class PointTransformerLayer(nn.Module):
    """
    Vector self-attention из Point Transformer (Zhao et al. 2021).
    Использует relative positional encoding δ и поэлементный attention.

    Формула:
        y_i = Σ_j softmax_j(γ(φ(x_i) − ψ(x_j) + δ(p_i−p_j))) ⊙ (α(x_j) + δ(p_i−p_j))

    Softmax вычисляется в FP32 чтобы избежать переполнения FP16 (exp(x) → Inf при x > ~10).
    """

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k = k
        # линейные проекции query / key / value
        self.phi   = nn.Linear(channels, channels, bias=False)
        self.psi   = nn.Linear(channels, channels, bias=False)
        self.alpha = nn.Linear(channels, channels, bias=False)
        # positional encoding: R^3 → C, с BN для стабильности
        self.delta = _MLP(3, channels, channels)
        # attention weight function: C → C (vector attention), с BN для стабильности
        self.gamma = _MLP(channels, channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, C) — входные признаки
            pos: (B, N, 3) — XYZ координаты
            idx: (B, N, K) — kNN-индексы (предвычисленные)
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        K = idx.shape[2]

        # признаки и позиции соседей
        idx_flat = idx.reshape(B, N * K)
        x_j   = _index_points(x,   idx_flat).view(B, N, K, C)   # (B,N,K,C)
        pos_j = _index_points(pos, idx_flat).view(B, N, K, 3)   # (B,N,K,3)

        # относительное позиционное кодирование
        pos_rel = pos.unsqueeze(2) - pos_j          # (B,N,K,3)
        delta   = self.delta(pos_rel)               # (B,N,K,C)

        # проекции
        q = self.phi(x).unsqueeze(2)               # (B,N,1,C)
        k = self.psi(x_j)                          # (B,N,K,C)
        v = self.alpha(x_j)                        # (B,N,K,C)

        # vector attention: softmax по оси K, поэлементно по C
        # ВАЖНО: явный cast в FP32 — при FP16 exp(x) переполняется при x > ~10,
        # что даёт Inf → NaN в softmax
        attn_input = self.gamma(q - k + delta)                    # (B,N,K,C)
        attn = torch.softmax(attn_input.float(), dim=2)           # FP32
        attn = attn.to(x.dtype)                                   # обратно в dtype модели

        return (attn * (v + delta)).sum(dim=2)                    # (B,N,C)


class PointTransformerBlock(nn.Module):
    """PT Layer с pre-BN и residual-connection."""

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k   = k
        self.bn  = nn.BatchNorm1d(channels)
        self.pt  = PointTransformerLayer(channels, k=k)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        # BN применяется к (B*N, C), затем форма восстанавливается
        x_norm = _apply_bn(x, self.bn)
        return x + self.pt(x_norm, pos, idx)


# ── Transition Down / Up ─────────────────────────────────────────────────────

class TransitionDown(nn.Module):
    """
    FPS-даунсэмплинг + kNN max-pool + линейная проекция.
    Уменьшает число точек N → npoint.
    """

    def __init__(self, in_channels: int, out_channels: int, npoint: int, k: int = 16,
                 fast_knn: bool = False):
        super().__init__()
        self.npoint   = npoint
        self.k        = k
        self.fast_knn = fast_knn
        self.bn_in  = nn.BatchNorm1d(in_channels)
        # mlp применяется к 2D тензору (B*npoint, in_channels) → BN1d работает напрямую
        self.mlp    = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:   (B, N, C)
            pos: (B, N, 3)
        Returns:
            x_new:   (B, npoint, out_channels)
            pos_new: (B, npoint, 3)
            idx:     (B, npoint, k) — kNN среди новых точек (для PT Block)
        """
        B, N, C = x.shape
        M = self.npoint

        # FPS для выбора репрезентативных точек
        fps_idx  = _farthest_point_sample(pos, M)                 # (B, M)
        pos_new  = _index_points(pos, fps_idx)                    # (B, M, 3)

        # нормализуем входные фичи перед пулингом
        x_norm = _apply_bn(x, self.bn_in)                        # (B, N, C)

        # kNN из оригинального облака → пулинг (max)
        pool_idx  = _knn_query(self.k, pos, pos_new, fast=self.fast_knn)   # (B, M, k)
        pool_flat = pool_idx.reshape(B, M * self.k)
        x_neigh   = _index_points(x_norm, pool_flat).view(B, M, self.k, C)
        x_new     = x_neigh.max(dim=2)[0]                                  # (B, M, C)

        # проекция: reshape в 2D для BN1d, затем обратно
        x_new = self.mlp(x_new.reshape(B * M, C)).reshape(B, M, -1)

        # kNN среди выбранных точек (используется в PT Block на этом уровне)
        k_self = min(self.k, M)
        idx    = _knn_query(k_self, pos_new, pos_new, fast=self.fast_knn)  # (B, M, k_self)

        return x_new, pos_new, idx


class TransitionUp(nn.Module):
    """
    Интерполяция (IDW, 3 ближайших) + skip-connection + линейная проекция.
    Восстанавливает число точек M → N.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        x_skip: torch.Tensor,
        pos_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, M, C)  — признаки низкого разрешения (source)
            pos:      (B, M, 3)  — позиции source
            x_skip:   (B, N, Cs) — skip-connection (target resolution)
            pos_skip: (B, N, 3)  — позиции target
        Returns:
            (B, N, out_channels)
        """
        B, N, _ = pos_skip.shape
        _, M, C = x.shape

        # inverse distance weighting от 3 ближайших точек source
        dist, idx = _square_distance(pos_skip.float(), pos.float()).topk(
            3, dim=-1, largest=False
        )
        dist = dist.clamp(min=1e-10)
        w    = 1.0 / dist
        w    = w / w.sum(dim=-1, keepdim=True)                       # (B, N, 3)

        x_interp = (_index_points(x, idx) * w.to(x.dtype).unsqueeze(-1)).sum(2)  # (B, N, C)

        out = torch.cat([x_interp, x_skip], dim=-1)                 # (B, N, C+Cs)
        # mlp применяется к 2D тензору — BN1d работает напрямую
        return self.mlp(out.reshape(B * N, -1)).reshape(B, N, -1)


# ── Segmentation Model ───────────────────────────────────────────────────────

class PointTransformerSegmentation(nn.Module):
    """
    Point Transformer U-Net для семантической сегментации.

    Encoder (5 стадий, FPS-даунсэмплинг):
        4096 → 1024 → 256 → 64 → 16 точек
        каналы: [c0, c1, c2, c3, c4]
    Decoder (4 стадии, IDW-интерполяция + skip):
        16 → 64 → 256 → 1024 → 4096 точек
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int = 9,
        k: int = 16,
        channels: tuple = (32, 64, 128, 256, 512),
        fast_knn: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fast_knn    = fast_knn
        c0, c1, c2, c3, c4 = channels

        # stem: входная проекция признаков
        self.stem_fc = nn.Linear(num_features, c0, bias=False)
        self.stem_bn = nn.BatchNorm1d(c0)

        # ── encoder ──────────────────────────────────────
        self.enc0 = PointTransformerBlock(c0, k=k)
        self.td1  = TransitionDown(c0, c1, npoint=1024, k=k, fast_knn=fast_knn)
        self.enc1 = PointTransformerBlock(c1, k=k)
        self.td2  = TransitionDown(c1, c2, npoint=256,  k=k, fast_knn=fast_knn)
        self.enc2 = PointTransformerBlock(c2, k=k)
        self.td3  = TransitionDown(c2, c3, npoint=64,   k=k, fast_knn=fast_knn)
        self.enc3 = PointTransformerBlock(c3, k=k)
        self.td4  = TransitionDown(c3, c4, npoint=16,   k=k, fast_knn=fast_knn)
        self.enc4 = PointTransformerBlock(c4, k=k)

        # ── decoder ──────────────────────────────────────
        self.tu3  = TransitionUp(c4, c3, c3)
        self.dec3 = PointTransformerBlock(c3, k=k)
        self.tu2  = TransitionUp(c3, c2, c2)
        self.dec2 = PointTransformerBlock(c2, k=k)
        self.tu1  = TransitionUp(c2, c1, c1)
        self.dec1 = PointTransformerBlock(c1, k=k)
        self.tu0  = TransitionUp(c1, c0, c0)
        self.dec0 = PointTransformerBlock(c0, k=k)

        # ── output head ───────────────────────────────────
        self.head = nn.Linear(c0, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, num_features) — облако точек
        Returns:
            (B, N, num_classes) — логиты per-point
        """
        B, N, nf = x.shape
        pos = x[:, :, :3].contiguous()  # XYZ координаты

        # stem: (B, N, nf) → (B, N, c0)
        x0 = F.relu(
            _apply_bn(self.stem_fc(x.reshape(B * N, nf)), self.stem_bn),
            inplace=True,
        ).reshape(B, N, -1)

        # enc0: kNN в исходном облаке
        idx0 = _knn_query(self.enc0.k, pos, pos, fast=self.fast_knn)
        x0   = self.enc0(x0, pos, idx0)                # (B, N, c0)

        # encoder
        x1, pos1, idx1 = self.td1(x0, pos)
        x1 = self.enc1(x1, pos1, idx1)                 # (B, 1024, c1)

        x2, pos2, idx2 = self.td2(x1, pos1)
        x2 = self.enc2(x2, pos2, idx2)                 # (B, 256, c2)

        x3, pos3, idx3 = self.td3(x2, pos2)
        x3 = self.enc3(x3, pos3, idx3)                 # (B, 64, c3)

        x4, pos4, idx4 = self.td4(x3, pos3)
        x4 = self.enc4(x4, pos4, idx4)                 # (B, 16, c4)

        # decoder
        d3  = self.tu3(x4, pos4, x3, pos3)
        idx = _knn_query(self.dec3.k, pos3, pos3, fast=self.fast_knn)
        d3  = self.dec3(d3, pos3, idx)                 # (B, 64, c3)

        d2  = self.tu2(d3, pos3, x2, pos2)
        idx = _knn_query(self.dec2.k, pos2, pos2, fast=self.fast_knn)
        d2  = self.dec2(d2, pos2, idx)                 # (B, 256, c2)

        d1  = self.tu1(d2, pos2, x1, pos1)
        idx = _knn_query(self.dec1.k, pos1, pos1, fast=self.fast_knn)
        d1  = self.dec1(d1, pos1, idx)                 # (B, 1024, c1)

        d0  = self.tu0(d1, pos1, x0, pos)
        idx = _knn_query(self.dec0.k, pos, pos, fast=self.fast_knn)
        d0  = self.dec0(d0, pos, idx)                  # (B, N, c0)

        return self.head(d0)                            # (B, N, num_classes)

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


# ── Classification Model ─────────────────────────────────────────────────────

class PointTransformerClassification(nn.Module):
    """
    Point Transformer для классификации облаков точек.
    Encoder-only: global max + avg pooling → MLP.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int = 9,
        k: int = 16,
        channels: tuple = (32, 64, 128, 256, 512),
        dropout: float = 0.5,
        fast_knn: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fast_knn = fast_knn
        c0, c1, c2, c3, c4 = channels

        self.stem_fc = nn.Linear(num_features, c0, bias=False)
        self.stem_bn = nn.BatchNorm1d(c0)

        self.enc0 = PointTransformerBlock(c0, k=k)
        self.td1  = TransitionDown(c0, c1, npoint=1024, k=k, fast_knn=fast_knn)
        self.enc1 = PointTransformerBlock(c1, k=k)
        self.td2  = TransitionDown(c1, c2, npoint=256,  k=k, fast_knn=fast_knn)
        self.enc2 = PointTransformerBlock(c2, k=k)
        self.td3  = TransitionDown(c2, c3, npoint=64,   k=k, fast_knn=fast_knn)
        self.enc3 = PointTransformerBlock(c3, k=k)
        self.td4  = TransitionDown(c3, c4, npoint=16,   k=k, fast_knn=fast_knn)
        self.enc4 = PointTransformerBlock(c4, k=k)

        self.head = nn.Sequential(
            nn.Linear(c4 * 2, 512, bias=False),
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
        B, N, nf = x.shape
        pos = x[:, :, :3].contiguous()

        x0 = F.relu(
            _apply_bn(self.stem_fc(x.reshape(B * N, nf)), self.stem_bn),
            inplace=True,
        ).reshape(B, N, -1)

        x0 = self.enc0(x0, pos, _knn_query(self.enc0.k, pos, pos, fast=self.fast_knn))

        x1, pos1, idx1 = self.td1(x0, pos)
        x1 = self.enc1(x1, pos1, idx1)

        x2, pos2, idx2 = self.td2(x1, pos1)
        x2 = self.enc2(x2, pos2, idx2)

        x3, pos3, idx3 = self.td3(x2, pos2)
        x3 = self.enc3(x3, pos3, idx3)

        x4, pos4, idx4 = self.td4(x3, pos3)
        x4 = self.enc4(x4, pos4, idx4)              # (B, 16, c4)

        x_max = x4.max(dim=1)[0]                     # (B, c4)
        x_avg = x4.mean(dim=1)                        # (B, c4)
        return self.head(torch.cat([x_max, x_avg], dim=-1))

    def get_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_weights=None,
    ) -> tuple:
        loss = F.cross_entropy(predictions, targets, weight=class_weights)
        return loss, loss, torch.tensor(0.0, device=predictions.device)
