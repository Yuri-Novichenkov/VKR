import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    T-Net для выравнивания входных данных
    """
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        # MLP для извлечения признаков
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)
        
        self.register_buffer('identity', torch.eye(k).unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: (B, k, N) - входные данные 
        Returns:
            transform: (B, k, k) - матрица трансформации
        """
        batch_size = x.size(0)
        
        # Извлечение признаков
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Полносвязные слои
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Формирование матрицы трансформации
        transform = x.view(batch_size, self.k, self.k)
        
        # Добавление единичной матрицы для стабильности
        transform = transform + self.identity
        
        return transform


class PointNetSegmentation(nn.Module):
    """
    PointNet
    """
    def __init__(self, num_classes, num_features=9):
        """
        Args:
            num_classes: количество классов для сегментации
            num_features: количество признаков на точку (X, Y, Z, R, G, B, Intensity)
        """
        super(PointNetSegmentation, self).__init__()
        
        self.num_classes = num_classes
        self.num_features = num_features
        
        # T-Net для трансформации координат
        self.input_transform = TNet(k=3)
        
        # Первый блок MLP для обработки координат
        self.conv1 = nn.Conv1d(num_features, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # T-Net для трансформации признаков
        self.feature_transform = TNet(k=64)
        
        # Второй блок MLP
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.conv4 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        
        # Блоки для сегментации объединение локальных и глобальных признаков
        self.conv5 = nn.Conv1d(1088, 512, 1)  # 64 + 1024 = 1088
        self.conv6 = nn.Conv1d(512, 256, 1)
        self.conv7 = nn.Conv1d(256, 128, 1)
        self.conv8 = nn.Conv1d(128, num_classes, 1)
        
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(256)
        self.bn7 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Args:
            x: (B, N, F) - батч облаков точек
               B - размер батча
               N - количество точек
               F - количество признаков (X, Y, Z, R, G, B, Intensity)
        Returns:
            logits: (B, N, num_classes) - логиты для каждого класса для каждой точки
            transform: (B, 3, 3) - матрица трансформации координат (для регуляризации)
        """
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Транспонирование для работы с Conv1d: (B, N, F) -> (B, F, N)
        x = x.transpose(2, 1)
        
        # Извлечение координат (первые 3 признака)
        coords = x[:, :3, :]
        
        # Трансформация координат
        transform_coords = self.input_transform(coords)
        coords = coords.transpose(2, 1)  # (B, N, 3)
        coords = torch.bmm(coords, transform_coords)  # Применение трансформации
        coords = coords.transpose(2, 1)  # (B, 3, N)
        
        # Объединение трансформированных координат с остальными признаками
        x = torch.cat([coords, x[:, 3:, :]], dim=1)
        
        # Первый блок MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Трансформация признаков
        transform_features = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform_features)
        x = x.transpose(2, 1)
        
        # Сохранение локальных признаков для последующего объединения
        local_features = x
        
        # Второй блок MLP
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Max pooling для получения глобальных признаков
        global_features = torch.max(x, 2, keepdim=True)[0]  # (B, 1024, 1)
        global_features = global_features.repeat(1, 1, num_points)  # (B, 1024, N)
        
        # Объединение локальных и глобальных признаков
        x = torch.cat([local_features, global_features], dim=1)  # (B, 1088, N)
        
        # Блоки для сегментации
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = self.conv8(x)  # (B, num_classes, N)
        
        # Транспонирование обратно: (B, num_classes, N) -> (B, N, num_classes)
        x = x.transpose(2, 1)
        
        return x, transform_coords, transform_features
    
    def get_loss(self, predictions, targets, transform_coords, transform_features, lambda_reg=0.001):
        """
        Вычисление функции потерь с регуляризацией для трансформаций
        
        Args:
            predictions: (B, N, num_classes) - предсказания модели
            targets: (B, N) - истинные метки классов
            transform_coords: (B, 3, 3) - матрица трансформации координат
            transform_features: (B, 64, 64) - матрица трансформации признаков
            lambda_reg: коэффициент регуляризации
        Returns:
            loss: общая функция потерь
        """
        # Основная функция потерь Cross Entropy
        # Изменяем форму: (B, N, num_classes) -> (B*N, num_classes) и (B, N) -> (B*N,)
        B, N, num_classes = predictions.shape

        predictions_flat = predictions.reshape(-1, num_classes)
        targets_flat = targets.reshape(-1)
        ce_loss = F.cross_entropy(predictions_flat, targets_flat)
        
        # Регуляризация трансформаций
        I = torch.eye(3, device=transform_coords.device).unsqueeze(0)
        reg_coords = torch.mean(torch.norm(
            torch.bmm(transform_coords, transform_coords.transpose(2, 1)) - I, dim=(1, 2)
        ))
        
        I = torch.eye(64, device=transform_features.device).unsqueeze(0)
        reg_features = torch.mean(torch.norm(
            torch.bmm(transform_features, transform_features.transpose(2, 1)) - I, dim=(1, 2)
        ))
        
        total_loss = ce_loss + lambda_reg * (reg_coords + reg_features)
        
        return total_loss, ce_loss, reg_coords + reg_features
