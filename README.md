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

### Установка PyTorch на GPU-сервере

`torch` не зафиксирован в `requirements.txt`, потому что способ установки зависит от ОС, драйвера и CUDA-окружения сервера.

Для Linux/GPU сначала установите обычные зависимости:

```bash
pip install -r requirements.txt
```

Затем отдельно установите PyTorch по официальной инструкции для вашей конфигурации CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Если используется готовый GPU-образ Selectel, перед установкой PyTorch рекомендуется проверить:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```
### Обучение модели
С рекомендуемыми параметрами
**PointNet (сегментация):**
```bash
python scripts/train.py --model pointnet --task segmentation --dataset Mar16 --amp --num_points 4096 --batch_size 8 --lr 0.001 --epochs 80 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```
```bash
python scripts/train.py --model pointnet --task segmentation --dataset Mar16 --num_points 4096 --batch_size 8 --lr 0.001 --epochs 100 --lambda_reg 0.001 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```
```bash
python scripts/test.py --checkpoint checkpoints/pointnet/segmentation/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 4096 --batch_size 4 --device cuda
```
```bash
python scripts/predictions.py --checkpoint checkpoints/pointnet/segmentation/mar16/best_model.pth --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 4096 --batch_size 4 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```
**PointNet++ (сегментация):**
```bash
python scripts/train.py --model pointnet++ --task segmentation --dataset Mar16 --amp --num_points 4096 --batch_size 4 --lr 0.001 --epochs 100 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```
```bash
python scripts/train.py --model pointnet++ --task segmentation --dataset Mar16 --num_points 4096 --batch_size 4 --lr 0.0005 --epochs 120 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```
```bash
python scripts/test.py --checkpoint checkpoints/pointnet++/segmentation/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 4096 --batch_size 4 --device cuda
```

**DGCNN (сегментация):**
```bash
python scripts/train.py --model dgcnn --task segmentation --dataset Mar16 --amp --k 8 --num_points 2048 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --batch_size 2
```
```bash
python scripts/test.py --checkpoint checkpoints/dgcnn/segmentation/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 4096 --batch_size 4 --device cuda
```

**LDGCNN (сегментация):**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --amp --num_points 2048 --batch_size 2 --k_small 8 --k_large 16 --lr 0.001 --epochs 80 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```
```bash
python scripts/test.py --checkpoint checkpoints/ldgcnn/segmentation/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 4096 --batch_size 4 --device cuda
```

### Обучение с class_balance (100 эпох, сегментация)
Рекомендуемый режим балансировки: `--class_balance effective --class_balance_beta 0.999`.

**PointNet + class_balance**
```bash
python scripts/train.py --model pointnet --task segmentation --dataset Mar16 --num_points 4096 --batch_size 8 --lr 0.001 --epochs 100 --lambda_reg 0.001 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**PointNet++ + class_balance**
```bash
python scripts/train.py --model pointnet++ --task segmentation --dataset Mar16 --num_points 4096 --batch_size 4 --lr 0.0005 --epochs 100 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**DGCNN + class_balance**
```bash
python scripts/train.py --model dgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --k 16 --lr 0.001 --epochs 100 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**LDGCNN + class_balance**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --k_small 12 --k_large 24 --lr 0.0008 --epochs 100 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

### Attention эксперименты для LDGCNN (E1/E2/E3)
Ниже готовые команды для запуска с балансировкой классов (`class_balance`).
Чекпоинты для разных режимов (`attention`, `loss_type`, `class_balance`) теперь автоматически сохраняются в отдельные подпапки внутри `checkpoints/<model>/<task>/...`.

**E1: LDGCNN + GATv2-style attention + class_balance**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --lr 0.0008 --epochs 100 --k_small 12 --k_large 24 --attention_type gatv2 --attention_k 16 --attention_heads 4 --attention_dropout 0.1 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**E2: LDGCNN + Local Window attention + class_balance**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --lr 0.0008 --epochs 100 --k_small 12 --k_large 24 --attention_type local_window --attention_k 8 --attention_heads 4 --attention_dropout 0.1 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**E3: LDGCNN контрольный запуск (без attention) + class_balance**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --lr 0.0008 --epochs 100 --k_small 12 --k_large 24 --attention_type none --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**Опционально для E1/E2/E3: CB-Focal loss**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --num_points 2048 --batch_size 4 --lr 0.0008 --epochs 100 --k_small 12 --k_large 24 --attention_type gatv2 --attention_k 16 --attention_heads 4 --attention_dropout 0.1 --loss_type cb_focal --focal_gamma 2.0 --class_balance effective --class_balance_beta 0.999 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512 --seed 42
```

**Тест для attention-экспериментов:**
```bash
python scripts/test.py --checkpoint checkpoints/ldgcnn/segmentation/attn_gatv2_k16_h4_d0p1/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 2048 --batch_size 4 --device cuda
```
```bash
python scripts/test.py --checkpoint checkpoints/ldgcnn/segmentation/attn_local_window_k8_h4_d0p1/mar16/best_model.pth --test_data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --num_points 2048 --batch_size 4 --device cuda
```

### Классификация облаков (все модели)
Для задачи `classification` доступны `pointnet`, `pointnet++`, `dgcnn`, `ldgcnn`.

**PointNet (классификация):**
```bash
python scripts/train.py --model pointnet --task classification --dataset Mar16 --num_points 4096 --batch_size 8 --lr 0.001 --epochs 80 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**PointNet++ (классификация):**
```bash
python scripts/train.py --model pointnet++ --task classification --dataset Mar16 --num_points 4096 --batch_size 8 --lr 0.001 --epochs 100 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**DGCNN (классификация):**
```bash
python scripts/train.py --model dgcnn --task classification --dataset Mar16 --num_points 2048 --batch_size 8 --lr 0.001 --epochs 100 --k 8 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**LDGCNN (классификация):**
```bash
python scripts/train.py --model ldgcnn --task classification --dataset Mar16 --num_points 2048 --batch_size 8 --lr 0.001 --epochs 100 --k_small 8 --k_large 16 --cache_dir cache --cache_mode read --cache_chunked --chunk_size 512
```

**Для генерации кэша:**
```bash
python scripts/train.py --dataset Mar16 --num_points 4096 --cache_dir cache --cache_mode write --cache_chunked --chunk_size 512 --cache_only
```
**Для визуализации файлов:**
```bash
python scripts/vizualization.py --data Files/Mar16/LiDAR/Mar16_test.txt --predictions predictions\pointnet\Mar16_test_GroundTruth\Mar16_test_GroundTruth_predictions.txt --color_by pred --max_points 1000000
```
```bash
python scripts/vizualization.py --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --color_by gt --max_points 1000000
```
```bash
python scripts/vizualization.py --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz --color_by rgb --max_points 1000000
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
- `--experiment_name`: имя эксперимента в MLflow (по умолчанию `PointCloudExperiments`)
- `--run_name`: имя запуска в MLflow (опционально; если не задано, формируется автоматически)
- `--attention_type`: attention-режим для `ldgcnn` (`none`, `gatv2`, `local_window`)
- `--attention_k`: размер локального окна attention
- `--attention_heads`: число attention-heads
- `--attention_dropout`: dropout attention
- `--loss_type`: тип loss (`ce`, `focal`, `cb_focal`)
- `--focal_gamma`: gamma для focal loss (по умолчанию `2.0`)
- `--class_balance`: балансировка классов в `CrossEntropy` (`none`, `inverse`, `effective`)
- `--class_balance_beta`: параметр beta для `effective` режима (по умолчанию `0.999`)

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

## MLflow

Логи экспериментов пишутся локально в `mlruns/`.
Пример запуска:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Рекомендуемая схема именования экспериментов

Чтобы в MLflow было проще анализировать результаты ВКР, рекомендуется делить запуски на смысловые серии:

- `VKR_Baselines` — базовые модели (`pointnet`, `pointnet++`, `dgcnn`, `ldgcnn`)
- `VKR_ClassBalance` — эксперименты с `--class_balance`
- `VKR_LDGCNN_Attention` — attention-эксперименты `E1/E2/E3`
- `VKR_Final` — финальные ("парадные") запуски для отчета
- `VKR_Ablation` — абляции (например `no-attention`, `no-class-balance`, `k12-24`)

### Формат `run_name`

Для обучения (`scripts/train.py`) рекомендуется формат:

`<model>__<task>__<dataset>__<variant>__pts<num_points>__bs<batch_size>__seed<seed>`

Примеры:

- `pointnet__segmentation__mar16__baseline__pts4096__bs8__seed42`
- `pointnet++__segmentation__mar16__cb-effective__pts4096__bs4__seed42`
- `ldgcnn__segmentation__mar16__attn-gatv2__pts2048__bs4__seed42`
- `ldgcnn__segmentation__mar16__attn-local-window__cb-effective__pts2048__bs4__seed42`

Для теста (`scripts/test.py`) рекомендуется формат:

`<model>__<task>__test__<dataset>__<variant>__<checkpoint>`

Примеры:

- `pointnet__segmentation__test__mar16__baseline__best_model`
- `ldgcnn__segmentation__test__mar16__attn-gatv2__best_model`

### Варианты `variant`

- `baseline` — базовый запуск без дополнительных методов
- `cb-effective`, `cb-inverse` — балансировка классов
- `attn-gatv2`, `attn-local-window` — attention-конфигурации
- `focal`, `cb-focal` — варианты функции потерь
- `no-attention`, `no-class-balance`, `k12-24` — абляции

Пример команды:

```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 --experiment_name VKR_LDGCNN_Attention --run_name "ldgcnn__segmentation__mar16__attn-gatv2__cb-effective__pts2048__bs4__seed42" --num_points 2048 --batch_size 4 --k_small 12 --k_large 24 --attention_type gatv2 --attention_k 16 --attention_heads 4 --attention_dropout 0.1 --class_balance effective --class_balance_beta 0.999 --seed 42
```

## Схемы архитектур (torchviz + hiddenlayer)

Для генерации схем добавлен скрипт `scripts/visualize_models.py`.

Установка зависимостей:
```bash
py -3 -m pip install torchviz hiddenlayer matplotlib graphviz
```

Запуск (все модели, сегментация):
```bash
py -3 scripts/visualize_models.py --models all --task segmentation --num_points 1024 --num_features 9 --num_classes 8 --output_dir diagrams
```

Запуск (одна модель, классификация):
```bash
py -3 scripts/visualize_models.py --models ldgcnn --task classification --num_points 1024 --num_features 9 --num_classes 8 --output_dir diagrams --attention_type gatv2
```

Примечание: для PNG/SVG нужен установленный системный Graphviz (`dot` в `PATH`).
Если `dot` не найден, скрипт все равно сохранит `.dot` файлы в `diagrams/`.

## Sweep batch-size (YAML) + сравнительные графики

Для автоматического сравнения нескольких `batch_size` добавлены скрипты:

- `scripts/run_sweep.py` — запускает серию прогонов `train.py` по YAML-конфигу.
- `scripts/plot_sweep_results.py` — строит графики из `runs_summary.csv`.

Папки результатов создаются с timestamp и не перезаписываются:

`experiments/batch_size/<timestamp>_<experiment>_<model>_<dataset>_<run_group>/`

Внутри каждой sweep-папки:

- `config_resolved.yaml` — сохранённый конфиг запуска
- `runs_summary.csv` — сводная таблица по всем `batch_size`
- `logs/` — логи каждого запуска
- `metrics/` — JSON-метрики каждого запуска
- `plots/` — сравнительные графики

### Пример YAML-конфига

Готовый пример:

`configs/sweeps/ldgcnn_gatv2_batch_sweep.yaml`

Также добавлены готовые конфиги:

- `configs/sweeps/pointnet_batch_sweep.yaml`
- `configs/sweeps/pointnetpp_batch_sweep.yaml`
- `configs/sweeps/dgcnn_batch_sweep.yaml`
- `configs/sweeps/ldgcnn_gatv2_batch_sweep.yaml`
- `configs/sweeps/ldgcnn_localwin_batch_sweep.yaml`
- `configs/sweeps/ldgcnn_none_batch_sweep.yaml`

### Запуск sweep

```bash
python scripts/run_sweep.py --config configs/sweeps/ldgcnn_gatv2_batch_sweep.yaml --output_root experiments/batch_size
```

По умолчанию `run_sweep.py` после завершения автоматически строит графики в `plots/`.
Отключить можно флагом `--no_auto_plot`.

Если нужно явно указать интерпретатор для дочерних запусков:

```bash
python scripts/run_sweep.py --config configs/sweeps/ldgcnn_gatv2_batch_sweep.yaml --python_executable .venv/bin/python
```

### Построение графиков

```bash
python scripts/plot_sweep_results.py --summary_csv experiments/batch_size/<your_sweep_folder>/runs_summary.csv
```

### Что логируется для сравнения

- `best_val_metric`
- `final_val_miou` / `final_val_accuracy`
- `mean_epoch_time_sec`
- `total_train_time_sec`
- `max_peak_vram_mb`
- `samples_per_sec` и `points_per_sec` (вычисляются из числа объектов/точек за эпоху)

VRAM берётся из PyTorch (`torch.cuda.max_memory_allocated`) и пишется по эпохам в MLflow/JSON.

## Литература

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)
- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)


## Автор

Новиченков Ю. Д. - ВКР 2025-2026
