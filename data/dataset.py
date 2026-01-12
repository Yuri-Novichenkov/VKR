import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import random


class LiDARDataset(Dataset):
    def __init__(self, data_path, num_points=4096, use_features=None, augment=False, has_labels=True):
        """
        Args:
            data_path: путь к txt файлу с данными
            num_points: количество точек в облаке
            use_features: список признаков для использования (None = все)
            augment: применять ли аугментацию данных
            has_labels: есть ли в данных колонка с метками (Classification)
        """
        self.data_path = Path(data_path)
        self.num_points = num_points
        self.augment = augment
        self.has_labels = has_labels

        print(f"Загрузка данных из {data_path}")
        self.data = pd.read_csv(data_path, sep='\t')
        print(f"Загружено {len(self.data)} точек")
        
        # Определение признаков
        feature_columns = ['X', 'Y', 'Z', 'R', 'G', 'B', 'Intensity', 
                          'NumberOfReturns', 'ReturnNumber']
        
        if use_features is None:
            self.use_features = feature_columns
        else:
            self.use_features = [f for f in use_features if f in feature_columns]
        
        # Извлечение признаков
        self.features = self.data[self.use_features].values.astype(np.float32)
        
        # Извлечение меток
        if has_labels and 'Classification' in self.data.columns:
            self.labels = self.data['Classification'].values.astype(np.int64)
        else:
            # фиктивные метки (все нули) для тестового набора без меток
            self.labels = np.zeros(len(self.features), dtype=np.int64)
            if has_labels:
                print("используются фиктивные метки")
        
        # Нормализация координат (центрирование и масштабирование)
        self._normalize_coords()
        
        # Нормализация остальных признаков
        self._normalize_features()
        
        # Получение уникальных классов и создание маппинга
        if has_labels and 'Classification' in self.data.columns:
            self.classes = np.unique(self.labels)
            self.num_classes = len(self.classes)
            
            # Создание маппинга классов на последовательные индексы 
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            
            # Применение маппинга к меткам
            self.labels = np.array([self.class_to_idx[cls] for cls in self.labels], dtype=np.int64)
        else:
            # Для тестового набора без меток используем фиктивные значения
            self.classes = np.array([0])
            self.num_classes = 1  # Будет переопределено при загрузке модели
            self.class_to_idx = {0: 0}
            self.idx_to_class = {0: 0}
        
        print(f"Количество классов: {self.num_classes}")
        print(f"Исходные классы: {self.classes}")
        print(f"Маппинг классов: {self.class_to_idx}")
        print(f"Используемые признаки: {self.use_features}")
        
        # Разбиение на облака точек
        self._create_point_clouds()
        
    def _normalize_coords(self):
        """Нормализация координат X, Y, Z"""
        if 'X' in self.use_features:
            x_idx = self.use_features.index('X')
            y_idx = self.use_features.index('Y')
            z_idx = self.use_features.index('Z')
            
            # Центрирование
            mean = np.mean(self.features[:, [x_idx, y_idx, z_idx]], axis=0)
            self.features[:, x_idx] -= mean[0]
            self.features[:, y_idx] -= mean[1]
            self.features[:, z_idx] -= mean[2]
            
            # Масштабирование
            std = np.std(self.features[:, [x_idx, y_idx, z_idx]], axis=0)
            std = np.where(std == 0, 1, std)  # Избегаем деления на ноль
            self.features[:, x_idx] /= std[0]
            self.features[:, y_idx] /= std[1]
            self.features[:, z_idx] /= std[2]
    
    def _normalize_features(self):
        """Нормализация остальных признаков (R, G, B, Intensity и т.д.)"""
        for i, feature_name in enumerate(self.use_features):
            # Пропускаем координаты
            if feature_name in ['X', 'Y', 'Z']:
                continue
            
            # Нормализация к диапазону [0, 1]
            feature_values = self.features[:, i]
            
            # Для цветов (R, G, B) - нормализация к [0, 1]
            if feature_name in ['R', 'G', 'B']:
                max_val = np.max(feature_values)
                if max_val > 0:
                    self.features[:, i] = feature_values / max_val
            
            # Для Intensity и других стандартизация
            elif feature_name in ['Intensity', 'NumberOfReturns', 'ReturnNumber']:
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                if std > 0:
                    self.features[:, i] = (feature_values - mean) / std
    
    def _create_point_clouds(self):
        """
        Разбиение данных на облака точек фиксированного размера
        """
        total_points = len(self.features)
        
        # разбиение на непересекающиеся блоки
        self.cloud_indices = []
        
        if total_points < self.num_points:
            # Если точек меньше чем нужно дублируем
            num_clouds = 1
            self.cloud_indices = [np.arange(total_points)]
        else:
            # Разбиваем на облака с перекрытием для увеличения количества данных
            step = self.num_points // 2  # 50% перекрытие
            num_clouds = (total_points - self.num_points) // step + 1
            
            for i in range(num_clouds):
                start_idx = i * step
                end_idx = start_idx + self.num_points
                if end_idx <= total_points:
                    self.cloud_indices.append(np.arange(start_idx, end_idx))
        
        print(f"Создано {len(self.cloud_indices)} облаков точек")
    
    def __len__(self):
        return len(self.cloud_indices)
    
    def __getitem__(self, idx):
        """
        Возвращает одно облако точек
        """
        # Получение индексов точек для этого облака
        point_indices = self.cloud_indices[idx]
        
        # Если точек меньше чем нужно дополняем случайными точками
        if len(point_indices) < self.num_points:
            # Дублируем случайные точки
            additional_indices = np.random.choice(point_indices, 
                                                 self.num_points - len(point_indices),
                                                 replace=True)
            point_indices = np.concatenate([point_indices, additional_indices])
        
        # Если точек больше выбираем случайную подвыборку
        elif len(point_indices) > self.num_points:
            point_indices = np.random.choice(point_indices, self.num_points, replace=False)
        
        # Извлечение признаков и меток
        features = self.features[point_indices]
        labels = self.labels[point_indices]
        
        # Аугментация данных
        if self.augment:
            features = self._augment_point_cloud(features)
        
        features = features.astype(np.float32)
        labels = labels.astype(np.int64)
        
        # Преобразование в тензоры с правильным типом
        features = torch.from_numpy(features).float()  
        labels = torch.from_numpy(labels).long() 
        
        return features, labels
    
    def _augment_point_cloud(self, points):
        """
        Аугментация облака точек
        """
        # Случайное вращение вокруг оси Z
        if random.random() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            
            # Применение вращения к координатам
            if 'X' in self.use_features and 'Y' in self.use_features and 'Z' in self.use_features:
                x_idx = self.use_features.index('X')
                y_idx = self.use_features.index('Y')
                z_idx = self.use_features.index('Z')
                coords = points[:, [x_idx, y_idx, z_idx]]
                coords = coords @ rotation_matrix.T
                points[:, x_idx] = coords[:, 0]
                points[:, y_idx] = coords[:, 1]
                points[:, z_idx] = coords[:, 2]
        
        # Случайное масштабирование
        if random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            if 'X' in self.use_features:
                points[:, self.use_features.index('X')] *= scale
            if 'Y' in self.use_features:
                points[:, self.use_features.index('Y')] *= scale
            if 'Z' in self.use_features:
                points[:, self.use_features.index('Z')] *= scale
        
        # Добавление шума
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, points.shape).astype(np.float32)
            points = points + noise
        
        return points.astype(np.float32)

