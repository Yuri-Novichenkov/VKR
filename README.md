# PointNet для семантической сегментации 3D-точечных облаков

Реализация модели PointNet для семантической сегментации 3D-точечных облаков местности на основе данных LiDAR.

**Тема ВКР:** Исследование и разработка нейросетевой модели для семантической сегментации 3D-точечных облаков местности на примере Hessigheim 3D Benchmark Dataset.

## Структура проекта

```
.
├── src/
│   ├── models/              # Модели (PointNet, PointNet++, DGCNN, LDGCNN)
│   ├── data/                # Загрузчики данных
│   └── utils/               # Общие операции (kNN, EdgeConv)
├── scripts/                 # Скрипты (train/test/visualize и др.)
├── notebooks/               # Jupyter notebooks
├── Files/                   # Данные (Mar16, Mar18)
├── requirements.txt         # Зависимости проекта
└── README.md               
```

## Установка
1. 
```bash
pip install -r requirements.txt
```
### Обучение модели
С рекомендуемыми параметрами
**PointNet (сегментация):**
```bash
python scripts/train.py --model pointnet --task segmentation --dataset Mar16 --amp \
  --num_points 4096 --batch_size 8 \
  --lr 0.001 --epochs 80 \
  --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**PointNet++ (сегментация):**
```bash
python scripts/train.py --model pointnet++ --task segmentation --dataset Mar16 --amp \
  --num_points 4096 --batch_size 4 \
  --lr 0.001 --epochs 100 \
  --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**DGCNN (сегментация):**
```bash
python scripts/train.py --model dgcnn --task segmentation --dataset Mar16 --amp --k 8 --num_points 2048 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --batch_size 2
```

**LDGCNN (сегментация):**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --amp --num_points 2048 --batch_size 2 --k_small 8 --k_large 16 --lr 0.001 --epochs 80 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**Для генерации кэша:**
```bash
python scripts/train.py --dataset Mar16 --num_points 4096 --cache_dir cache --cache_mode write --cache_chunked --chunk_size 512 --cache_only
```

### Параметры обучения

- `--train_data`: путь к обучающему набору данных (опционально)
- `--val_data`: путь к валидационному набору данных (опционально)
- `--data_root`: корень данных, например `Files/Mar18/LiDAR`
- `--dataset`: префикс датасета (`Mar16` или `Mar18`)
- `--num_points`: количество точек в облаке (по умолчанию: 4096)
- `--batch_size`: размер батча (по умолчанию: 8)
- `--epochs`: количество эпох (по умолчанию: 100)
- `--lr`: скорость обучения (по умолчанию: 0.001)
- `--lambda_reg`: коэффициент регуляризации трансформаций (по умолчанию: 0.001)
- `--save_dir`: директория для сохранения моделей (по умолчанию: `checkpoints`)
- `--resume`: путь к чекпоинту для возобновления обучения (опционально)
- `--model`: модель (`pointnet`, `pointnet++`, `dgcnn`, `ldgcnn`)
- `--task`: задача (`segmentation` или `classification`)

## Формат данных
- `X`, `Y`, `Z`: координаты точек
- `R`, `G`, `B`: цвет точек
- `Intensity`: интенсивность
- `NumberOfReturns`, `ReturnNumber`: информация о возвратах лидара
- `Classification`: метка класса для каждой точки

## Данные

Данные лежат в `Files/`:
- `Files/Mar16/LiDAR/Mar16_train.txt`
- `Files/Mar16/LiDAR/Mar16_val.txt`
- `Files/Mar16/LiDAR/Mar16_test.txt`
- `Files/Mar18/LiDAR/Mar18_train.txt`
- `Files/Mar18/LiDAR/Mar18_val.txt`
- `Files/Mar18/LiDAR/Mar18_test.txt`


## Возобновление обучения


```bash
python scripts/train.py --resume checkpoints/last_checkpoint.pth
```

Все результаты будут сохранены в директории `visualizations/`.

## MLflow

Логи экспериментов пишутся локально в `mlruns/`.
Пример запуска:
```bash
python scripts/train.py --model dgcnn --task segmentation --dataset Mar18 --experiment_name PointCloudExperiments
```

## Литература

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)


## Автор

Новиченков Ю. Д. - ВКР 2024-2025

