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
├── requirements.txt         # Зависимости проекта
└── README.md               
```

## Установка
1. 
```bash
pip install -r requirements.txt
```
### Обучение модели

```bash
python train.py --train_data LiDAR/Mar16_train.txt --val_data LiDAR/Mar16_val.txt
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

## Модель

Модель PointNet включает:
- **T-Net (Transformation Network)**: выравнивание входных координат и признаков
- **MLP блоки**: извлечение признаков из точек
- **Max Pooling**: агрегация глобальных признаков
- **Сегментационный блок**: предсказание класса для каждой точки

## Метрики

Во время обучения отслеживаются следующие метрики:
- **Loss**: общая функция потерь (Cross Entropy + регуляризация)
- **Accuracy**: точность классификации
- **mIoU (mean Intersection over Union)**: средний IoU по всем классам
- **Per-class IoU**: IoU для каждого класса отдельно

## Сохранение моделей

Модели сохраняются в директории `checkpoints`:
- `best_model.pth`: лучшая модель по валидационному mIoU
- `last_checkpoint.pth`: последний чекпоинт для возобновления обучения

## Возобновление обучения

Для возобновления обучения с сохраненного чекпоинта:

```bash
python train.py --resume checkpoints/last_checkpoint.pth
```

## Сохранение предсказаний

Для сохранения предсказаний модели на тестовом наборе в файл:

```bash
python save_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --test_data LiDAR/Mar16_test.txt \
    --output predictions.txt
```

Это создаст файл с исходными данными и добавленной колонкой `Predicted_Classification`.

## Визуализация результатов

Для создания графиков и визуализаций результатов обучения:

```bash
python visualize.py \
    --checkpoint checkpoints/best_model.pth \
    --test_data LiDAR/Mar16_test.txt \
    --predictions predictions.txt \
    --output_dir visualizations
```

Это создаст:
- Графики обучения (loss, accuracy, mIoU)
- Confusion matrix
- График IoU по классам
- Распределение классов
- Визуализацию облаков точек
- Текстовый отчет с метриками

Все результаты будут сохранены в директории `visualizations/`.

## Структура данных

Данные должны быть в формате txt с разделителем табуляции. Структура файлов:
- `LiDAR/Mar16_train.txt` - обучающий набор
- `LiDAR/Mar16_val.txt` - валидационный набор  
- `LiDAR/Mar16_test.txt` - тестовый набор

## Результаты

- **Валидационный mIoU:** 56.02%
- **Accuracy на валидации:** 88.14%
- **Количество классов:** 11
- **Параметров модели:** 3,533,204

## Литература

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)

## Автор

Новиченков Ю. Д. - ВКР 2024-2025

