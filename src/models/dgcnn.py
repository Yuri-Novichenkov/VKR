import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.point_ops import get_graph_feature, EdgeConv, knn


class DGCNNSegmentation(nn.Module):
    """
    DGCNN для семантической сегментации
    """
    def __init__(self, num_classes, num_features=9, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.num_features = num_features

        self.ec1 = EdgeConv(in_channels=2 * num_features, out_channels=64)
        self.ec2 = EdgeConv(in_channels=2 * 64, out_channels=64)
        self.ec3 = EdgeConv(in_channels=2 * 64, out_channels=128)
        self.ec4 = EdgeConv(in_channels=2 * 128, out_channels=256)

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
        # x: (B, N, F)
        x = x.transpose(2, 1).contiguous()  # (B, F, N)

        x1 = self.ec1(get_graph_feature(x, k=self.k))
        x2 = self.ec2(get_graph_feature(x1, k=self.k))
        x3 = self.ec3(get_graph_feature(x2, k=self.k))
        x4 = self.ec4(get_graph_feature(x3, k=self.k))

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x = self.conv1(x_cat)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)  # (B, num_classes, N)
        x = x.transpose(2, 1).contiguous()  # (B, N, num_classes)
        return x

    def get_loss(self, predictions, targets):
        B, N, num_classes = predictions.shape
        predictions_flat = predictions.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        loss = F.cross_entropy(predictions_flat, targets_flat)
        return loss, loss, torch.tensor(0.0, device=predictions.device)


class DGCNNClassification(nn.Module):
    """
    DGCNN для классификации облаков точек
    """
    def __init__(self, num_classes, num_features=9, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.num_features = num_features

        self.ec1 = EdgeConv(in_channels=2 * num_features, out_channels=64)
        self.ec2 = EdgeConv(in_channels=2 * 64, out_channels=64)
        self.ec3 = EdgeConv(in_channels=2 * 64, out_channels=128)
        self.ec4 = EdgeConv(in_channels=2 * 128, out_channels=256)

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
        # x: (B, N, F)
        x = x.transpose(2, 1).contiguous()  # (B, F, N)

        x1 = self.ec1(get_graph_feature(x, k=self.k))
        x2 = self.ec2(get_graph_feature(x1, k=self.k))
        x3 = self.ec3(get_graph_feature(x2, k=self.k))
        x4 = self.ec4(get_graph_feature(x3, k=self.k))

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x = self.conv1(x_cat)  # (B, emb_dims, N)

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
