import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LiDARDataset(Dataset):
    def __init__(
        self,
        data_path,
        num_points=4096,
        use_features=None,
        augment=False,
        has_labels=True,
        task="segmentation",
        cache_dir=None,
        cache_mode="write",
        cache_chunked=False,
        chunk_size=512,
    ):
        """
        Args:
            data_path: путь к txt файлу с данными
            num_points: количество точек в облаке
            use_features: список признаков для использования (None = все)
            augment: применять ли аугментацию данных
            has_labels: есть ли в данных колонка с метками (Classification)
            task: segmentation | classification
        """
        if task not in ("segmentation", "classification"):
            raise ValueError("task должен быть 'segmentation' или 'classification'")

        self.data_path = Path(data_path)
        self.num_points = num_points
        self.augment = augment
        self.has_labels = has_labels
        self.task = task
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_mode = cache_mode
        self.cache_chunked = cache_chunked
        self.chunk_size = int(chunk_size)
        self._chunk_cache = None
        self._chunk_cache_idx = None

        if self.cache_mode not in ("off", "read", "write"):
            raise ValueError("cache_mode должен быть 'off', 'read' или 'write'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size должен быть > 0")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_cache_path(use_features)
        if self.cache_mode in ("read", "write") and cache_path:
            chunk_manifest = self._get_chunk_manifest_path(cache_path)
            if self.cache_chunked and chunk_manifest.exists():
                self._load_cache_chunked(chunk_manifest)
                return
            if cache_path.exists():
                self._load_cache(cache_path)
                return

        print(f"Загрузка данных из {data_path}")
        self.data = pd.read_csv(data_path, sep="\t")
        print(f"Загружено {len(self.data)} точек")
        
        # Определение признаков
        feature_columns = ["X", "Y", "Z", "R", "G", "B", "Intensity", "NumberOfReturns", "ReturnNumber"]
        available_columns = set(self.data.columns)
        
        if use_features is None:
            requested_features = feature_columns
        else:
            requested_features = [f for f in use_features if f in feature_columns]

        self.use_features = [f for f in requested_features if f in available_columns]
        missing = [f for f in requested_features if f not in available_columns]
        if missing:
            print(f"Предупреждение: отсутствуют признаки в данных: {missing}")

        if not all(f in self.use_features for f in ["X", "Y", "Z"]):
            raise ValueError("В данных должны быть координаты X, Y, Z")
        
        # Извлечение признаков
        self.features = self.data[self.use_features].values.astype(np.float32)
        
        # Извлечение меток
        if has_labels and "Classification" in self.data.columns:
            self.labels = self.data["Classification"].values.astype(np.int64)
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
        if has_labels and "Classification" in self.data.columns:
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

        if self.cache_mode == "write" and cache_path:
            if self.cache_chunked:
                self._save_cache_chunked(cache_path)
            else:
                self._save_cache(cache_path)
        
    def _normalize_coords(self):
        """Нормализация координат X, Y, Z"""
        if "X" in self.use_features:
            x_idx = self.use_features.index("X")
            y_idx = self.use_features.index("Y")
            z_idx = self.use_features.index("Z")
            
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
            if feature_name in ["X", "Y", "Z"]:
                continue
            
            # Нормализация к диапазону [0, 1]
            feature_values = self.features[:, i]
            
            # Для цветов (R, G, B) - нормализация к [0, 1]
            if feature_name in ["R", "G", "B"]:
                max_val = np.max(feature_values)
                if max_val > 0:
                    self.features[:, i] = feature_values / max_val
            
            # Для Intensity и других стандартизация
            elif feature_name in ["Intensity", "NumberOfReturns", "ReturnNumber"]:
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                if std > 0:
                    self.features[:, i] = (feature_values - mean) / std
    
    def _create_point_clouds(self):
        """
        Разбиение данных на облака точек фиксированного размера
        """
        total_points = len(self.features)
        
        if total_points < self.num_points:
            # Если точек меньше чем нужно дополняем с повторением
            indices = np.random.choice(total_points, self.num_points, replace=True)
            self.cloud_indices = indices.reshape(1, -1).astype(np.int32)
        else:
            # Разбиваем на облака с перекрытием для увеличения количества данных
            step = self.num_points // 2  # 50% перекрытие
            num_clouds = (total_points - self.num_points) // step + 1
            starts = np.arange(num_clouds, dtype=np.int64) * step
            self.cloud_indices = (
                starts[:, None] + np.arange(self.num_points, dtype=np.int64)[None, :]
            ).astype(np.int32)
        
        print(f"Создано {len(self.cloud_indices)} облаков точек")
    
    def __len__(self):
        if getattr(self, "chunked", False):
            return int(self.num_clouds)
        return len(self.cloud_indices)
    
    def __getitem__(self, idx):
        """
        Возвращает одно облако точек
        """
        if getattr(self, "chunked", False):
            cloud_features, cloud_labels = self._load_chunk_item(idx)
            features = cloud_features
            labels = cloud_labels
        else:
            # Получение индексов точек для этого облака
            point_indices = self.cloud_indices[idx]

            # Извлечение признаков и меток
            features = self.features[point_indices]
            labels = self.labels[point_indices]
        
        # Аугментация данных
        if self.augment:
            features = self._augment_point_cloud(features)
        
        features = features.astype(np.float32)
        labels = labels.astype(np.int64)
        
        # Преобразование в тензоры
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        if self.task == "classification":
            # Облачный класс как мода по точкам
            labels_np = labels.numpy()
            cloud_label = int(np.bincount(labels_np).argmax())
            return features, torch.tensor(cloud_label, dtype=torch.long)
        
        return features, labels

    def _get_cache_path(self, use_features):
        if not self.cache_dir:
            return None

        key = {
            "data_path": str(self.data_path.resolve()),
            "num_points": int(self.num_points),
            "use_features": use_features,
            "has_labels": bool(self.has_labels),
            "task": self.task,
            "cache_chunked": bool(self.cache_chunked),
            "chunk_size": int(self.chunk_size),
        }
        key_json = json.dumps(key, sort_keys=True)
        digest = hashlib.md5(key_json.encode("utf-8")).hexdigest()
        filename = f"{self.data_path.stem}_n{self.num_points}_{digest}.npz"
        return self.cache_dir / filename

    def _save_cache(self, cache_path):
        metadata = {
            "use_features": self.use_features,
            "classes": self.classes.tolist() if isinstance(self.classes, np.ndarray) else list(self.classes),
            "class_to_idx": {str(k): int(v) for k, v in self.class_to_idx.items()},
            "idx_to_class": {str(k): int(v) for k, v in self.idx_to_class.items()},
            "num_classes": int(self.num_classes),
        }
        np.savez(
            cache_path,
            features=self.features,
            labels=self.labels,
            cloud_indices=self.cloud_indices,
            metadata=json.dumps(metadata),
        )
        print(f"Кэш сохранен: {cache_path}")

    def _load_cache(self, cache_path):
        print(f"Загрузка кэша: {cache_path}")
        with np.load(cache_path, allow_pickle=True) as data:
            self.features = data["features"]
            self.labels = data["labels"]
            self.cloud_indices = data["cloud_indices"]
            metadata = json.loads(str(data["metadata"]))

        self.use_features = metadata.get("use_features", [])
        self.classes = np.array(metadata.get("classes", []))
        self.class_to_idx = {int(k): int(v) for k, v in metadata.get("class_to_idx", {}).items()}
        self.idx_to_class = {int(k): int(v) for k, v in metadata.get("idx_to_class", {}).items()}
        self.num_classes = int(metadata.get("num_classes", 1))

        print(f"Кэш загружен. Используемые признаки: {self.use_features}")

    def _get_chunk_manifest_path(self, cache_path):
        return cache_path.with_suffix(".chunks.json")

    def _save_cache_chunked(self, cache_path):
        chunk_manifest = self._get_chunk_manifest_path(cache_path)
        chunk_dir = self.cache_dir / f"{self.data_path.stem}_chunks_{cache_path.stem}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        num_clouds = len(self.cloud_indices)
        chunk_files = []
        for start in range(0, num_clouds, self.chunk_size):
            end = min(start + self.chunk_size, num_clouds)
            idx_slice = self.cloud_indices[start:end]
            cloud_features = self.features[idx_slice]
            cloud_labels = self.labels[idx_slice]
            chunk_file = chunk_dir / f"chunk_{start:06d}_{end:06d}.npz"
            np.savez(chunk_file, features=cloud_features, labels=cloud_labels)
            chunk_files.append(chunk_file.name)

        metadata = {
            "chunk_dir": str(chunk_dir),
            "chunk_files": chunk_files,
            "chunk_size": int(self.chunk_size),
            "num_clouds": int(num_clouds),
            "num_points": int(self.num_points),
            "num_features": int(self.features.shape[1]),
            "use_features": self.use_features,
            "classes": self.classes.tolist() if isinstance(self.classes, np.ndarray) else list(self.classes),
            "class_to_idx": {str(k): int(v) for k, v in self.class_to_idx.items()},
            "idx_to_class": {str(k): int(v) for k, v in self.idx_to_class.items()},
            "num_classes": int(self.num_classes),
        }
        chunk_manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Кэш чанков сохранен: {chunk_manifest}")

    def _load_cache_chunked(self, chunk_manifest):
        print(f"Загрузка кэша чанков: {chunk_manifest}")
        metadata = json.loads(chunk_manifest.read_text(encoding="utf-8"))
        self.chunked = True
        self.chunk_dir = Path(metadata["chunk_dir"])
        self.chunk_files = list(metadata["chunk_files"])
        self.chunk_size = int(metadata["chunk_size"])
        self.num_clouds = int(metadata["num_clouds"])
        self.num_points = int(metadata["num_points"])
        self.use_features = metadata.get("use_features", [])
        self.classes = np.array(metadata.get("classes", []))
        self.class_to_idx = {int(k): int(v) for k, v in metadata.get("class_to_idx", {}).items()}
        self.idx_to_class = {int(k): int(v) for k, v in metadata.get("idx_to_class", {}).items()}
        self.num_classes = int(metadata.get("num_classes", 1))
        self.features = None
        self.labels = None
        print(f"Кэш чанков загружен. Используемые признаки: {self.use_features}")

    def _load_chunk_item(self, idx):
        chunk_idx = idx // self.chunk_size
        local_idx = idx % self.chunk_size

        if self._chunk_cache is None or self._chunk_cache_idx != chunk_idx:
            chunk_file = self.chunk_dir / self.chunk_files[chunk_idx]
            with np.load(chunk_file, allow_pickle=False) as data:
                self._chunk_cache = (data["features"], data["labels"])
            self._chunk_cache_idx = chunk_idx

        cloud_features = self._chunk_cache[0][local_idx]
        cloud_labels = self._chunk_cache[1][local_idx]
        return cloud_features, cloud_labels
    
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
                [0, 0, 1],
            ])
            
            # Применение вращения к координатам
            if "X" in self.use_features and "Y" in self.use_features and "Z" in self.use_features:
                x_idx = self.use_features.index("X")
                y_idx = self.use_features.index("Y")
                z_idx = self.use_features.index("Z")
                coords = points[:, [x_idx, y_idx, z_idx]]
                coords = coords @ rotation_matrix.T
                points[:, x_idx] = coords[:, 0]
                points[:, y_idx] = coords[:, 1]
                points[:, z_idx] = coords[:, 2]
        
        # Случайное масштабирование
        if random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            if "X" in self.use_features:
                points[:, self.use_features.index("X")] *= scale
            if "Y" in self.use_features:
                points[:, self.use_features.index("Y")] *= scale
            if "Z" in self.use_features:
                points[:, self.use_features.index("Z")] *= scale
        
        # Добавление шума
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, points.shape).astype(np.float32)
            points = points + noise
        
        return points.astype(np.float32)
