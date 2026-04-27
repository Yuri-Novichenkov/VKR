"""
Dataset для LiDAR облаков точек.

Поддерживает:
- чтение .txt/.las/.laz;
- нормализацию признаков;
- разбиение на окна фиксированного размера;
- аугментации;
- кэширование (монолитное и чанковое);
- два режима задачи: segmentation / classification.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import load_dataframe as _load_dataframe_shared


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
        class_to_idx=None,
        normalize_stats=None,
    ):
        """
        Args:
            data_path: путь к txt файлу с данными
            num_points: количество точек в облаке
            use_features: список признаков для использования (None = все)
            augment: применять ли аугментацию данных
            has_labels: есть ли в данных колонка с метками (Classification)
            task: segmentation | classification
            normalize_stats: статистики z-score из train-набора (dict или None).
                Если None — вычисляются из текущих данных (train-поведение).
                Если dict — применяются переданные stats (val/test).
                После инициализации доступны как self.normalize_stats.
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
        self.external_class_to_idx = (
            {int(cls): int(idx) for cls, idx in class_to_idx.items()}
            if class_to_idx is not None
            else None
        )
        self.external_normalize_stats = normalize_stats
        self.normalize_stats = {}
        self._chunk_cache = None
        self._chunk_cache_idx = None

        if self.cache_mode not in ("off", "read", "write"):
            raise ValueError("cache_mode должен быть 'off', 'read' или 'write'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size должен быть > 0")

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_cache_path(use_features)
        # Если кэш найден, загружаем подготовленные тензоры и пропускаем
        # тяжелый этап парсинга/нормализации/разбиения.
        if self.cache_mode in ("read", "write") and cache_path:
            chunk_manifest = self._get_chunk_manifest_path(cache_path)
            if self.cache_chunked and chunk_manifest.exists():
                self._load_cache_chunked(chunk_manifest)
                return
            if cache_path.exists():
                self._load_cache(cache_path)
                return

        print(f"Загрузка данных из {data_path}")
        self.data = _load_dataframe_shared(self.data_path)
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

        # Контракт для графовых моделей: первые 3 канала всегда XYZ.
        # Это гарантирует корректность kNN по геометрии (x[:, :3, :]) в
        # DGCNN/LDGCNN и единообразный порядок входа для всех моделей.
        extra_features = [f for f in self.use_features if f not in ["X", "Y", "Z"]]
        self.use_features = ["X", "Y", "Z"] + extra_features
        
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

        # Нормализация признаков: для Intensity z-score берётся из train-stats,
        # чтобы val/test нормализовались на той же шкале, что и train.
        self.normalize_stats = self._normalize_features(stats=self.external_normalize_stats)
        
        # Получение уникальных классов и создание маппинга меток
        # в непрерывный диапазон [0, num_classes-1] для CrossEntropy.
        if has_labels and "Classification" in self.data.columns:
            self.classes = np.unique(self.labels)
            if self.external_class_to_idx is not None:
                missing_classes = [int(cls) for cls in self.classes if int(cls) not in self.external_class_to_idx]
                if missing_classes:
                    raise ValueError(
                        "В данных найдены классы, отсутствующие в переданном mapping: "
                        f"{missing_classes}"
                    )
                self.class_to_idx = dict(self.external_class_to_idx)
            else:
                # Создание маппинга классов на последовательные индексы
                self.class_to_idx = {int(cls): idx for idx, cls in enumerate(self.classes)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
            self.num_classes = len(self.class_to_idx)

            # Применение маппинга к меткам
            self.labels = np.array([self.class_to_idx[cls] for cls in self.labels], dtype=np.int64)
        else:
            # Для тестового набора без меток используем фиктивные значения
            self.classes = np.array([0])
            if self.external_class_to_idx is not None:
                self.class_to_idx = dict(self.external_class_to_idx)
                self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
                self.num_classes = len(self.class_to_idx)
            else:
                self.num_classes = 1  # Будет переопределено при загрузке модели
                self.class_to_idx = {0: 0}
                self.idx_to_class = {0: 0}
        
        print(f"Количество классов: {self.num_classes}")
        print(f"Исходные классы: {self.classes}")
        print(f"Маппинг классов: {self.class_to_idx}")
        print(f"Используемые признаки: {self.use_features}")
        
        # Разбиение на облака точек фиксированного размера.
        self._create_point_clouds()

        if self.cache_mode == "write" and cache_path:
            if self.cache_chunked:
                self._save_cache_chunked(cache_path)
            else:
                self._save_cache(cache_path)
        
    def _normalize_coords(self):
        """Глобальное центрирование координат X, Y, Z.

        Per-block нормализация (центрирование по центроиду окна + масштабирование
        к единичному шару) делается в `__getitem__`. Это важно для PointNet++
        (фиксированные радиусы ball query) и DGCNN/LDGCNN (локальные kNN):
        модели рассчитаны на то, что каждый блок ограничен единичной областью,
        а не z-score по всей сцене.
        """
        if "X" in self.use_features:
            x_idx = self.use_features.index("X")
            y_idx = self.use_features.index("Y")
            z_idx = self.use_features.index("Z")

            # Только центрирование сцены (для численной стабильности
            # при больших абсолютных UTM-координатах). Масштабирование
            # выполняется локально в __getitem__.
            mean = np.mean(self.features[:, [x_idx, y_idx, z_idx]], axis=0)
            self.features[:, x_idx] -= mean[0]
            self.features[:, y_idx] -= mean[1]
            self.features[:, z_idx] -= mean[2]
    
    def _normalize_features(self, stats=None):
        """
        Нормализация R/G/B/Intensity и т.д.
        stats=None → вычисляет z-score из текущих данных (train).
        stats=dict → применяет переданные статистики (val/test).
        Возвращает dict с использованными статистиками z-score признаков.
        """
        feat_stats = {}
        for i, feature_name in enumerate(self.use_features):
            if feature_name in ["X", "Y", "Z"]:
                continue

            feature_values = self.features[:, i]

            # RGB: фиксированный диапазон разрядности — не зависит от файла.
            # 16-bit (0–65535) или 8-bit (0–255) определяется по max_val.
            if feature_name in ["R", "G", "B"]:
                max_val = float(np.max(feature_values))
                if max_val > 255.0:
                    norm_range = 65535.0
                elif max_val > 1.0:
                    norm_range = 255.0
                else:
                    norm_range = 1.0
                self.features[:, i] = feature_values / norm_range

            # Intensity, NumberOfReturns, ReturnNumber: z-score.
            # Для val/test используем статистики из train, чтобы
            # нормализация была согласована между сплитами.
            elif feature_name in ["Intensity", "NumberOfReturns", "ReturnNumber"]:
                if stats and feature_name in stats:
                    mean = float(stats[feature_name]["mean"])
                    std = float(stats[feature_name]["std"])
                else:
                    mean = float(np.mean(feature_values))
                    std = float(np.std(feature_values))
                if std > 0:
                    self.features[:, i] = (feature_values - mean) / std
                feat_stats[feature_name] = {"mean": mean, "std": std}

        return feat_stats
    
    def _create_point_clouds(self):
        """
        Пространственная нарезка сцены на окна фиксированного размера с
        гарантированным 100% покрытием всех точек.

        Для каждой seed-точки выбирается `num_points` ближайших соседей по XYZ
        (kNN через KD-дерево). После первого прохода по случайным seed-ам
        докидываем дополнительные блоки для всех ещё непокрытых точек —
        это важно для корректного scene-level voting при оценке: если хотя бы
        одна точка не попала ни в одно окно, её метка недоступна модели,
        и метрики смещаются.
        """
        total_points = len(self.features)

        if total_points < self.num_points:
            # Гарантируем 100% покрытие: каждая исходная точка должна попасть
            # хотя бы в один блок. Поэтому сначала берём все индексы без
            # повторений, затем добиваем до num_points с replacement.
            base_idx = np.arange(total_points, dtype=np.int64)
            pad_count = self.num_points - total_points
            pad_idx = np.random.choice(total_points, pad_count, replace=True).astype(np.int64)
            indices = np.concatenate([base_idx, pad_idx], axis=0)
            np.random.shuffle(indices)
            self.cloud_indices = indices.reshape(1, -1).astype(np.int32)
            print(f"Создано {len(self.cloud_indices)} облаков точек (padding, coverage=100%)")
            return

        # Извлечение координат для пространственной нарезки
        x_idx = self.use_features.index("X")
        y_idx = self.use_features.index("Y")
        z_idx = self.use_features.index("Z")
        xyz = self.features[:, [x_idx, y_idx, z_idx]].astype(np.float64)

        # Построение KD-дерева (используется и для основного прохода, и для добора)
        tree = None
        backend = None
        query_fn = None
        try:
            from sklearn.neighbors import KDTree

            tree = KDTree(xyz)
            backend = "sklearn.KDTree"

            def _query(seeds_xyz):
                idx = tree.query(seeds_xyz, k=self.num_points, return_distance=False)
                return np.asarray(idx, dtype=np.int32)

            query_fn = _query
        except ImportError:
            pass

        if tree is None:
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(xyz)
                backend = "scipy.cKDTree"

                def _query(seeds_xyz):
                    _d, idx = tree.query(seeds_xyz, k=self.num_points)
                    return np.asarray(idx, dtype=np.int32)

                query_fn = _query
            except ImportError:
                pass

        if query_fn is None:
            backend = "numpy fallback"

            def _query(seeds_xyz):
                out = np.empty((len(seeds_xyz), self.num_points), dtype=np.int32)
                for i in range(len(seeds_xyz)):
                    diff = xyz - seeds_xyz[i]
                    d = np.einsum("ij,ij->i", diff, diff)
                    nn = np.argpartition(d, self.num_points - 1)[: self.num_points]
                    out[i] = nn
                return out

            query_fn = _query

        # Основной проход: случайные seed-ы, ~2x покрытие
        num_clouds = max(1, (2 * total_points + self.num_points - 1) // self.num_points)
        rng = np.random.default_rng(42)
        replace_seeds = num_clouds > total_points
        seed_indices = rng.choice(total_points, size=num_clouds, replace=replace_seeds)

        cloud_indices = query_fn(xyz[seed_indices])
        if cloud_indices.ndim == 1:
            cloud_indices = cloud_indices.reshape(1, -1)

        # Гарантируем полное покрытие: пока есть непокрытые точки,
        # используем их как seed для дополнительных блоков. Каждый новый блок
        # покрывает минимум 1 новую точку (сам seed) и ещё num_points-1 соседей,
        # поэтому цикл сходится за O(total_points / num_points) итераций.
        covered = np.zeros(total_points, dtype=bool)
        covered[cloud_indices.ravel()] = True
        extra_blocks = 0
        max_iterations = 64  # защита от бесконечного цикла
        iteration = 0
        while not covered.all() and iteration < max_iterations:
            iteration += 1
            uncovered_idx = np.flatnonzero(~covered)
            # Берём не более чем по 1 seed на (num_points/4) непокрытых точек,
            # чтобы быстро сокращать число итераций, не раздувая датасет.
            batch = max(1, min(len(uncovered_idx), len(uncovered_idx) // max(1, self.num_points // 4) + 1))
            # Выбираем равномерно по индексам непокрытых — этого достаточно,
            # потому что каждый добавленный блок уже kNN-образно тянет соседей.
            if batch >= len(uncovered_idx):
                extra_seed_idx = uncovered_idx
            else:
                step = max(1, len(uncovered_idx) // batch)
                extra_seed_idx = uncovered_idx[::step][:batch]

            extra_nn = query_fn(xyz[extra_seed_idx])
            if extra_nn.ndim == 1:
                extra_nn = extra_nn.reshape(1, -1)
            cloud_indices = np.concatenate([cloud_indices, extra_nn], axis=0)
            covered[extra_nn.ravel()] = True
            extra_blocks += len(extra_nn)

        if not covered.all():
            # Крайне маловероятно, но на всякий случай — жёсткая гарантия:
            # для каждой оставшейся точки создаём персональный блок.
            remaining = np.flatnonzero(~covered)
            extra_nn = query_fn(xyz[remaining])
            if extra_nn.ndim == 1:
                extra_nn = extra_nn.reshape(1, -1)
            cloud_indices = np.concatenate([cloud_indices, extra_nn], axis=0)
            covered[extra_nn.ravel()] = True
            extra_blocks += len(extra_nn)

        assert covered.all(), "Не удалось гарантировать 100% покрытие точек блоками"

        self.cloud_indices = cloud_indices.astype(np.int32)
        print(
            f"Создано {len(self.cloud_indices)} облаков точек "
            f"(основных={num_clouds}, добор={extra_blocks}, backend={backend})"
        )
    
    def __len__(self):
        if getattr(self, "chunked", False):
            return int(self.num_clouds)
        return len(self.cloud_indices)
    
    def __getitem__(self, idx):
        """
        Возвращает одно облако точек
        """
        if getattr(self, "chunked", False):
            cloud_features, cloud_labels, _cloud_indices = self._load_chunk_item(idx)
            features = cloud_features.copy()
            labels = cloud_labels
        else:
            # Получение индексов точек для этого облака
            point_indices = self.cloud_indices[idx]

            # Извлечение признаков и меток; .copy() важен, чтобы последующая
            # per-block нормализация и аугментация не затронули self.features.
            features = self.features[point_indices].copy()
            labels = self.labels[point_indices]

        # Per-block нормализация координат (центрирование по центроиду окна
        # и масштабирование к единичному шару). Приводит каждый блок к одному
        # масштабу, на который рассчитаны радиусы PointNet++ (0.2/0.4) и
        # kNN в DGCNN/LDGCNN.
        if "X" in self.use_features:
            x_idx = self.use_features.index("X")
            y_idx = self.use_features.index("Y")
            z_idx = self.use_features.index("Z")
            coords = features[:, [x_idx, y_idx, z_idx]].astype(np.float32)
            centroid = coords.mean(axis=0, keepdims=True)
            coords = coords - centroid
            max_norm = float(np.max(np.linalg.norm(coords, axis=1)))
            if max_norm > 1e-6:
                coords = coords / max_norm
            features[:, x_idx] = coords[:, 0]
            features[:, y_idx] = coords[:, 1]
            features[:, z_idx] = coords[:, 2]

        # Аугментация данных
        if self.augment:
            features = self._augment_point_cloud(features)
        
        features = features.astype(np.float32)
        labels = labels.astype(np.int64)
        
        # Преобразование в тензоры
        features = torch.from_numpy(features).float()
        labels = torch.from_numpy(labels).long()

        if self.task == "classification":
            # Для cloud-level классификации используем класс-моду по точкам окна.
            labels_np = labels.numpy()
            cloud_label = int(np.bincount(labels_np).argmax())
            return features, torch.tensor(cloud_label, dtype=torch.long)
        
        return features, labels

    def get_cloud_point_indices(self, idx):
        if getattr(self, "chunked", False):
            cloud_indices = self._load_chunk_item(idx)[2]
            return cloud_indices.astype(np.int64)
        return self.cloud_indices[idx].astype(np.int64)

    def _get_cache_path(self, use_features):
        if not self.cache_dir:
            return None

        key = {
            # cache_format=4: пространственная нарезка блоков с гарантией
            # полного покрытия + фиксированный порядок входных каналов
            # (XYZ первыми). Изменение ломает совместимость со
            # старыми кэшами, поэтому бампаем версию.
            "cache_format": 4,
            "data_path": str(self.data_path.resolve()),
            "num_points": int(self.num_points),
            "use_features": use_features,
            "has_labels": bool(self.has_labels),
            "task": self.task,
            "cache_chunked": bool(self.cache_chunked),
            "chunk_size": int(self.chunk_size),
            "class_to_idx": self.external_class_to_idx,
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
        # Чанковый кэш хранит уже нарезанные облака, чтобы быстро грузить
        # большие датасеты без пикового потребления RAM.
        for start in range(0, num_clouds, self.chunk_size):
            end = min(start + self.chunk_size, num_clouds)
            idx_slice = self.cloud_indices[start:end]
            cloud_features = self.features[idx_slice]
            cloud_labels = self.labels[idx_slice]
            chunk_file = chunk_dir / f"chunk_{start:06d}_{end:06d}.npz"
            np.savez(chunk_file, features=cloud_features, labels=cloud_labels, point_indices=idx_slice)
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
                self._chunk_cache = (data["features"], data["labels"], data["point_indices"])
            self._chunk_cache_idx = chunk_idx

        cloud_features = self._chunk_cache[0][local_idx]
        cloud_labels = self._chunk_cache[1][local_idx]
        cloud_indices = self._chunk_cache[2][local_idx]
        return cloud_features, cloud_labels, cloud_indices
    
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
        
        # Случайное масштабирование.
        # После масштабирования повторно нормализуем координаты к единичному
        # шару, чтобы не нарушать предположения ball query в PointNet++
        # (радиусы 0.2/0.4) и kNN-окрестности в DGCNN/LDGCNN.
        if random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            if all(ax in self.use_features for ax in ["X", "Y", "Z"]):
                x_idx = self.use_features.index("X")
                y_idx = self.use_features.index("Y")
                z_idx = self.use_features.index("Z")
                points[:, x_idx] *= scale
                points[:, y_idx] *= scale
                points[:, z_idx] *= scale
                # Повторная нормализация к единичному шару
                coords = points[:, [x_idx, y_idx, z_idx]]
                max_norm = float(np.max(np.linalg.norm(coords, axis=1)))
                if max_norm > 1e-6:
                    points[:, x_idx] = coords[:, 0] / max_norm
                    points[:, y_idx] = coords[:, 1] / max_norm
                    points[:, z_idx] = coords[:, 2] / max_norm
        
        # Добавление шума
        if random.random() > 0.5:
            coord_indices = [self.use_features.index(axis) for axis in ["X", "Y", "Z"] if axis in self.use_features]
            if coord_indices:
                noise = np.random.normal(0, 0.02, (points.shape[0], len(coord_indices))).astype(np.float32)
                points[:, coord_indices] = points[:, coord_indices] + noise

        return points.astype(np.float32)

    # Метод сохранён для обратной совместимости; делегирует в src.data.io.
    def _load_dataframe(self, data_path):
        return _load_dataframe_shared(data_path)
