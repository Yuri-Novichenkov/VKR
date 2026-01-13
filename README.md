# PointNet для семантической сегментации 3D-точечных облаков

Реализация модели PointNet для семантической сегментации 3D-точечных облаков местности на основе данных LiDAR.

**Тема ВКР:** Исследование и разработка нейросетевой модели для семантической сегментации 3D-точечных облаков местности на примере Hessigheim 3D Benchmark Dataset.

## Структура проекта

```
.
├── models/
│   └── pointnet.py          # Реализация модели PointNet
├── data/
│   └── dataset.py           # Загрузчик данных для txt файлов
├── train.py                 # Скрипт для обучения модели
├── compare_models.py        # Скрипт для сравнения моделей
├── requirements.txt         # Зависимости проекта
└── README.md               
```

## Установка
1. 
```bash
pip install -r requirements.txt
```
### Обучение модели

**PointNet:**
```bash
python train.py --train_data LiDAR/Mar16_train.txt --val_data LiDAR/Mar16_val.txt --model pointnet
```

**PointNet++:**
```bash
python train.py --train_data LiDAR/Mar16_train.txt --val_data LiDAR/Mar16_val.txt --model pointnet++
```

### Параметры обучения

- `--train_data`: путь к обучающему набору данных (по умолчанию: `LiDAR/Mar16_train.txt`)
- `--val_data`: путь к валидационному набору данных (по умолчанию: `LiDAR/Mar16_val.txt`)
- `--num_points`: количество точек в облаке (по умолчанию: 4096)
- `--batch_size`: размер батча (по умолчанию: 8)
- `--epochs`: количество эпох (по умолчанию: 100)
- `--lr`: скорость обучения (по умолчанию: 0.001)
- `--lambda_reg`: коэффициент регуляризации трансформаций (по умолчанию: 0.001)
- `--save_dir`: директория для сохранения моделей (по умолчанию: `checkpoints`)
- `--resume`: путь к чекпоинту для возобновления обучения (опционально)
- `--model`: модель для обучения - `pointnet` или `pointnet++` (по умолчанию: `pointnet`)
- `--model`: модель для обучения - `pointnet` или `pointnet++` (по умолчанию: `pointnet`)

### п ример

```bash
python train.py \
    --train_data LiDAR/Mar16_train.txt \
    --val_data LiDAR/Mar16_val.txt \
    --num_points 4096 \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.001 \
    --save_dir checkpoints
```

## Формат данных
- `X`, `Y`, `Z`: координаты точек
- `R`, `G`, `B`: цвет точек
- `Intensity`: интенсивность
- `NumberOfReturns`, `ReturnNumber`: информация о возвратах лидара
- `Classification`: метка класса для каждой точки


## Возобновление обучения


```bash
python train.py --resume checkpoints/last_checkpoint.pth
```

Все результаты будут сохранены в директории `visualizations/`.

## Результаты

### PointNet
- **Валидационный mIoU:** 56.02%
- **Accuracy на валидации:** 88.14%
- **Количество классов:** 11
- **Параметров модели:** 3,533,204

### PointNet++
- **Валидационный mIoU:** 51.74%
- **Accuracy на валидации:** 86.84%
- **Количество классов:** 11
- **Параметров модели:** 1,404,747
- **Эпоха лучшей модели:** 99

## Литература

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)


## Автор

Новиченков Ю. Д. - ВКР 2024-2025

