import torch


def pairwise_distance(x):
    """
    Вычисляет расстояния попарно.
    Args:
        x: (B, C, N)
    Returns:
        dist: (B, N, N)
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    dist = xx.transpose(2, 1) + inner + xx
    return dist


def knn(x, k):
    """
    KNN поиск
    Args:
        x: (B, C, N)
        k: число соседей
    Returns:
        idx: (B, N, k)
    """
    dist = pairwise_distance(x)
    _, idx = dist.topk(k=k, dim=-1, largest=False, sorted=True)
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Формирует edge features для EdgeConv.
    Args:
        x: (B, C, N)
        k: число соседей
        idx: (B, N, k) если уже посчитан
    Returns:
        edge_feature: (B, 2*C, N, k)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=k)  # (B, N, k)

    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    feature = x.view(B * N, -1)[idx, :]
    feature = feature.view(B, N, k, C)  # (B, N, k, C)
    x = x.view(B, N, 1, C).repeat(1, 1, k, 1)

    # edge = concat(x_j - x_i, x_i)
    edge_feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    return edge_feature


class EdgeConv(torch.nn.Module):
    """
    EdgeConv блок
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, x):
        # x: (B, in_channels, N, k)
        x = self.conv(x)
        x = x.max(dim=-1)[0]  # (B, out_channels, N)
        return x
