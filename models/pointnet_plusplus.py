import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet import TNet


def square_distance(src, dst):
    """
    Вычисление квадрата расстояния между двумя наборами точек
    Args:
        src: (B, N, C) - исходные точки
        dst: (B, M, C) - целевые точки
    Returns:
        dist: (B, N, M) - квадраты расстояний
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Индексирование точек по индексам
    Args:
        points: (B, N, C) - точки
        idx: (B, M) или (B, M, K) - индексы
    Returns:
        new_points: (B, M, C) или (B, M, K, C) - выбранные точки
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    выбор наиболее удаленных точек
    Args:
        xyz: (B, N, 3) - координаты точек
        npoint: количество точек для выборки
    Returns:
        centroids: (B, npoint) - индексы выбранных точек
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    группировка точек в шаре заданного радиуса
    Args:
        radius: радиус шара
        nsample: максимальное количество точек в группе
        xyz: (B, N, 3) - все точки
        new_xyz: (B, S, 3) - центроиды (выбранные точки)
    Returns:
        group_idx: (B, S, nsample) - индексы точек в группах
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Args:
        npoint: количество точек для выборки
        radius: радиус для группировки
        nsample: количество точек в каждой группе
        xyz: (B, N, 3) - координаты
        points: (B, N, C) - признаки точек
    Returns:
        new_xyz: (B, npoint, 3) - координаты выбранных точек
        new_points: (B, npoint, nsample, C+3) - признаки групп
    """
    B, N, C = xyz.shape
    S = npoint
    
    # Farthest Point Sampling
    fps_idx = farthest_point_sample(xyz, npoint)  # (B, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, npoint, 3)
    
    # Grouping
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, 3)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # Нормализация относительно центроида
    
    if points is not None:
        grouped_points = index_points(points, idx)  # (B, npoint, nsample, C)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # (B, npoint, nsample, C+3)
    else:
        new_points = grouped_xyz_norm  # (B, npoint, nsample, 3)
    
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Sampling и Grouping для всех точек последний слой
    """
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(xyz.device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    """
    основной блок PointNet++
    """
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Args:
            xyz: (B, N, 3) - координаты точек
            points: (B, N, C) - признаки точек
        Returns:
            new_xyz: (B, npoint, 3) - координаты выбранных точек
            new_points: (B, npoint, mlp[-1]) - новые признаки
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        
        # new_points: (B, npoint, nsample, C+3)
        new_points = new_points.permute(0, 3, 2, 1)  # (B, C+3, nsample, npoint)
        
        # Применение MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        
        # Max pooling по nsample
        new_points = torch.max(new_points, 2)[0]  # (B, mlp[-1], npoint)
        new_points = new_points.permute(0, 2, 1)  # (B, npoint, mlp[-1])
        
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """
    интерполяция признаков при upsampling
    """
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1: (B, N, 3) - координаты точек с большим разрешением
            xyz2: (B, M, 3) - координаты точек с меньшим разрешением
            points1: (B, N, C1) - признаки точек с большим разрешением
            points2: (B, M, C2) - признаки точек с меньшим разрешением
        Returns:
            new_points: (B, N, mlp[-1]) - интерполированные признаки
        """
        B, N, C = xyz1.shape
        _, M, _ = xyz2.shape

        if M == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # Берем 3 ближайшие точки

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)  # (B, C, N)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = new_points.permute(0, 2, 1)  # (B, N, mlp[-1])
        
        return new_points


class PointNetPlusPlusSegmentation(nn.Module):
    """
    PointNet++ для семантической сегментации
    """
    def __init__(self, num_classes, num_features=9, normal_channel=False):
        """
        Args:
            num_classes: количество классов
            num_features: количество признаков на точку
            normal_channel: использовать ли нормали (для координат)
        """
        super(PointNetPlusPlusSegmentation, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        
        # Извлечение координат и признаков
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        
        # Encoder 
        # sa1: в sample_and_group объединяются grouped_xyz_norm (3) + features (num_features - 3)
        feature_dim = (num_features - 3) if num_features > 3 else 0
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, 
                                         in_channel=3 + feature_dim,  # 3 координаты + признаки
                                         mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64,
                                         in_channel=128 + 3,
                                         mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                         in_channel=256 + 3, group_all=True,
                                         mlp=[256, 512, 1024])
        
        # Decoder (Feature Propagation Layers)
        # fp3: 256 (l2_points) + 1024 (l3_points) = 1280
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        # fp2: 128 (l1_points) + 256 (l2_points после fp3) = 384
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        # fp1: (num_features - 3) исходные признаки без координат + 128 (l1_points после fp2)
        # Если features = None, то только 128, иначе (num_features - 3) + 128
        feature_dim = (num_features - 3) if num_features > 3 else 0
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + feature_dim, mlp=[128, 128, 128])
        
        # Классификатор
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        """
        Args:
            x: (B, N, F) - батч облаков точек
        Returns:
            logits: (B, N, num_classes) - логиты для каждого класса
        """
        # Извлечение координат и признаков
        coords = x[:, :, :3]  # (B, N, 3)
        if x.shape[2] > 3:
            features = x[:, :, 3:]  # (B, N, F-3)
        else:
            features = None
        
        # Encoder
        l1_xyz, l1_points = self.sa1(coords, features)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Decoder
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(coords, l1_xyz, features, l1_points)
        
        # Классификатор
        feat = l0_points.permute(0, 2, 1)  # (B, 128, N)
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = self.drop1(feat)
        feat = self.conv2(feat)  # (B, num_classes, N)
        feat = feat.permute(0, 2, 1)  # (B, N, num_classes)
        
        return feat
    
    def get_loss(self, predictions, targets):
        """
        Args:
            predictions: (B, N, num_classes) - предсказания модели
            targets: (B, N) - истинные метки классов
        Returns:
            loss: функция потерь
        """
        B, N, num_classes = predictions.shape
        predictions_flat = predictions.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(predictions_flat, targets_flat)
        return loss, loss, torch.tensor(0.0)

