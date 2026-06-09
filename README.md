# Семантическая сегментация 3D-облаков точек LiDAR

**Тема ВКР:** Исследование и разработка нейросетевой модели для семантической сегментации 3D-облаков точек местности на примере датасета Hessigheim 3D Benchmark (H3D).

---

## Содержание

1. [Установка](#установка)
2. [Данные](#данные)
3. [Архитектуры моделей](#архитектуры-моделей)
4. [Обучение](#обучение)
5. [Тест и предсказания](#тест-и-предсказания)
6. [Визуализация](#визуализация)
7. [Дистилляция знаний (финал ВКР)](#дистилляция-знаний-финал-вкр)
8. [Оптимизация инференса](#оптимизация-инференса)
9. [Система экспериментов](#система-экспериментов)
10. [Результаты](#результаты)
11. [Литература](#литература)
12. [Автор](#автор)

---

## Установка

```bash
# 1. Зависимости проекта
pip install -r requirements.txt

# 2. Инструменты анализа (профилировщики, ONNX, torch_cluster)
pip install -r requirements-dev.txt

# 3. PyTorch устанавливается отдельно под вашу версию CUDA
#    https://pytorch.org/get-started/locally/
# Пример для CUDA 12.8 (A5000):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Проверка GPU-сервера перед установкой:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

**MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

---

## Данные

**Датасет:** Hessigheim 3D (H3D) — UAV LiDAR, 11 классов, ~1.4M точек на сплит.

| Файлы | Путь |
|-------|------|
| Mar16 (основной) | `Files/Mar16/LiDAR/` |
| Mar18 (кросс-датасет) | `Files/Mar18/LiDAR/` |
| Mar19 (кросс-датасет) | `Files/Mar19/LiDAR/` |

Формат файлов: `.laz` (приоритет) → `.las` → `.txt`. Тестовый файл с метками: `Mar16_test_GroundTruth.laz`.

**Входные признаки на точку (9 каналов):** X, Y, Z, R, G, B, Intensity, NumberOfReturns, ReturnNumber.

**11 классов Hessigheim:**
`0=Low Vegetation, 1=Impervious Surface, 2=Vehicle, 3=Urban Furniture, 4=Roof, 5=Facade, 6=Shrub, 7=Tree, 8=Soil/Gravel, 9=Vertical Surface, 10=Chimney`

---

## Архитектуры моделей

Все модели реализованы в `src/models/`, создаются через `build_model()` в `src/models/factory.py`.

| Модель | Класс | Mar16 mIoU | Параметры |
|--------|-------|-----------|-----------|
| `pointnet` | `PointNetSegmentation` | 42.47% | TNet + shared MLP |
| `pointnet++` | `PointNetPlusPlusSegmentation` | 49.15% | FPS + Set Abstraction + FP |
| `dgcnn` | `DGCNNSegmentation` | 53.13% | Dynamic graph EdgeConv (feature space) |
| `ldgcnn` (none) | `LDGCNNSegmentation` | 54.76% | EdgeConv в XYZ-пространстве |
| `ldgcnn` (local_window) | `LDGCNNSegmentation` | 55.05% | + local window attention |
| `ldgcnn` (gatv2) | `LDGCNNSegmentation` | 55.44% | + GATv2 attention |
| `ldgcnn_flash` (Full) | `LDGCNNFlashSegmentation` | 60.88% | LDGCNN + Flash Self-Attention (SDPA) |
| `ldgcnn_flash` (Small) | `LDGCNNFlashSegmentation` | ~ожидается | Каналы 32/64/128, 266K парам. |
| `pointtransformer` | `PointTransformerSegmentation` | **61.65%** | U-Net + vector self-attention |

**LDGCNNFlash** — авторская модель ВКР. Сочетает локальный граф EdgeConv из LDGCNN с глобальным контекстом через Flash Self-Attention (SDPA, PyTorch ≥ 2.0). Flash Attention использует tile-based вычисление: O(N·d) памяти без материализации матрицы N×N, ~3–4× быстрее наивного attention при FP16.

---

## Обучение

### Финальные команды (лучшие гиперпараметры из `configs/best_params.yaml`)

**LDGCNNFlash (Full) — лучший конфиг:**
```bash
python scripts/train.py --model ldgcnn_flash --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 16 --epochs 100 --lr 0.001 --lr_scheduler cosine \
  --amp --loss_type ce --class_balance effective --class_balance_beta 0.99999 --seed 42 \
  --cache_mode read --cache_chunked --chunk_size 512 --num_workers 4
```

**PointTransformer:**
```bash
python scripts/train.py --model pointtransformer --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 16 --epochs 100 --lr 0.001 --lr_scheduler cosine \
  --amp --loss_type lovasz --class_balance effective --class_balance_beta 0.99999 --seed 42 \
  --cache_mode read --cache_chunked --chunk_size 512 --num_workers 4
```

**LDGCNN + GATv2 (лучшая graph-модель):**
```bash
python scripts/train.py --model ldgcnn --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 16 --epochs 100 --lr 0.001 --lr_scheduler cosine \
  --amp --loss_type ce --class_balance effective --class_balance_beta 0.9999 --seed 42 \
  --attention_type gatv2 --attention_k 16 --attention_heads 4 --attention_dropout 0.1 \
  --cache_mode read --cache_chunked --chunk_size 512 --num_workers 4
```

**PointNet++ (AMP отключён — NaN при FP16):**
```bash
python scripts/train.py --model pointnet++ --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 16 --epochs 100 --lr 0.0003 --lr_scheduler cosine \
  --loss_type lovasz --class_balance effective --class_balance_beta 0.99999 --seed 42 \
  --cache_mode read --cache_chunked --chunk_size 512 --num_workers 4
```

### Подготовка кэша (разовая операция, ускоряет обучение)
```bash
python scripts/train.py --dataset Mar16 --num_points 4096 \
  --cache_dir cache --cache_mode write --cache_chunked --chunk_size 512 --cache_only
```

---

## Тест и предсказания

```bash
# Тест с scene-level voting (финальные метрики)
python scripts/test.py \
  --checkpoint checkpoints/loss_sweep/ldgcnn_flash/segmentation/cb_effective_b0p99999/mar16/best_model.pth \
  --dataset Mar16 --num_points 4096 --batch_size 8 --device cuda

# Предсказания на CPU-ноутбуке (~1–30 мин в зависимости от модели)
python scripts/predictions.py \
  --checkpoint checkpoints/loss_sweep/ldgcnn_flash/segmentation/cb_effective_b0p99999/mar16/best_model.pth \
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz \
  --output_root predictions --device cpu
```

Примерное время предсказаний на CPU (Mar16, 1239 облаков, N=4096):
PointNet ~1.5 мин, PointNet++ ~6 мин, LDGCNNFlash ~10 мин, LDGCNN ~28 мин.

---

## Визуализация

Open3D интерактивный просмотр + легенда (`figures/legend.png`):

```bash
python scripts/visualization.py \
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz \
  --predictions predictions/ldgcnn_flash/Mar16_test_GroundTruth/Mar16_test_GroundTruth_predictions.txt \
  --color_by pred --num_classes 11 \
  --class_config configs/classes/hessigheim.yaml \
  --legend --max_points 300000

# Ground Truth
python scripts/visualization.py 
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz 
  --color_by gt --num_classes 11 
  --class_config configs/classes/hessigheim.yaml --legend --max_points 300000

# RGB-раскраска
python scripts/visualization.py \
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz \
  --color_by rgb --max_points 300000
```

---

## Дистилляция знаний (финал ВКР)

**Идея:** сжать LDGCNNFlash (931K параметров, 60.88% mIoU) в LDGCNNFlash-Small (266K параметров, каналы 32/64/128 вместо 64/128/256), используя PointTransformer (61.65%) как учителя. Цель — получить модель с в ~3.5× меньшим числом параметров при сохранении mIoU ≥ 0.5.

**Формула KD loss:**
```
L = (1 - α) · L_task + α · T² · KL(student_logits/T ‖ teacher_logits/T)
```

### Параметры дистилляции

| Параметр | Значение | Обоснование |
|----------|----------|-------------|
| Учитель | PointTransformer (61.65%) | Лучшая модель; чекпоинт уже есть |
| Студент-архитектура | `flash_channels=[32, 64, 128]` | 3.5× меньше параметров (266K vs 931K) |
| `kd_alpha` | 0.5 | Равный баланс task loss и KD loss |
| `kd_temperature` | 4.0 | Стандартное значение (Hinton et al., 2015); сглаживает soft targets |
| Task loss | CE + CB-Loss (β=0.99999) | Лучший конфиг для LDGCNNFlash Full |
| LR, epochs | 0.001, 100 | Те же что у полной модели |
| Batch size | 16 | Та же что у полной модели |

**Почему temperature=4.0:** при T→1 soft targets близки к one-hot (нет передачи тёмных знаний), при T→∞ все классы одинаково вероятны. T=4 — стандарт литературы, который хорошо работает для задач сегментации с 11 классами.

**Почему α=0.5:** задача сегментации требует точного соответствия меткам (task loss важен). α=0.5 позволяет учителю влиять на обучение, не подавляя сигнал от ground truth.

### Три эксперимента для сравнения

| Run | Архитектура | KD | Ожидаемый результат |
|-----|------------|-----|-------------------|
| Small baseline | 32/64/128 (266K) | нет | Нижняя граница |
| Full + KD | 64/128/256 (931K) | от PT | Проверка: даёт ли KD прирост |
| **Small + KD** | 32/64/128 (266K) | от PT | **Ключевой результат** |

### Запуск

```bash
# Все три эксперимента последовательно (авто-resume при прерывании)
python scripts/run_sweep.py --config configs/sweeps/kd_ldgcnn_flash_small.yaml

# Возобновить только ключевой run
python scripts/run_sweep.py --config configs/sweeps/kd_ldgcnn_flash_small.yaml \
  --resume --only_model "LDGCNNFlashSmall+KD"

# Запуск вручную (Small + KD)
python scripts/train.py --model ldgcnn_flash --flash_channels 32 64 128 \
  --task segmentation --dataset Mar16 --num_points 4096 --batch_size 16 \
  --epochs 100 --lr 0.001 --lr_scheduler cosine --amp --seed 42 \
  --loss_type ce --class_balance effective --class_balance_beta 0.99999 \
  --kd_teacher_checkpoint checkpoints/loss_sweep/pointtransformer/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth \
  --kd_alpha 0.5 --kd_temperature 4.0 \
  --cache_mode read --cache_chunked --chunk_size 512 --num_workers 4
```

Результаты сохраняются в `experiments/distillation/`.

### Файлы на сервер (загрузить)

```
scripts/train.py                           # KD логика (load_teacher, kd_alpha, kd_temperature)
scripts/run_sweep.py                       # Поддержка режима runs: + list-аргументы
src/models/factory.py                      # flash_channels параметр
configs/sweeps/kd_ldgcnn_flash_small.yaml  # Конфиг 3 экспериментов
```

### Файлы с сервера (забрать)

```
experiments/distillation/                  # Логи, метрики, runs_summary.csv
checkpoints/distillation/                  # best_model.pth для каждого run-а
mlflow.db                                  # Обновлённая БД метрик
```

---

## Оптимизация инференса

### GPU (RTX A5000, N=4096, batch=1)

| Модель | Лучший режим | Время | Ускорение |
|--------|-------------|-------|-----------|
| PointNet | compile_fp16 | 1.38 мс | 1.8× |
| DGCNN | compile_fp16 | 3.7 мс | 3.9× |
| LDGCNNFlash | compile_fp16 | 5.9 мс | 2.5× |
| PointTransformer | compile_fp32 | 12.8 мс | **11.8×** |
| LDGCNN | compile_fp16 | 17.1 мс | 2.2× |
| PointNet++ | fp32 (без ускорения) | ~178 мс | — |

```bash
# Бенчмарк всех режимов
python scripts/benchmark_inference.py --all_mar16 \
  --modes fp32 fp16 compile_fp32 compile_fp16 --batch_size 1 \
  --output_csv results/benchmark_gpu.csv

# N-sensitivity (скорость + mIoU для N=512–4096)
python scripts/n_sensitivity.py

# k-sensitivity (граф-модели)
python scripts/k_sensitivity.py

# INT8 квантизация
python scripts/optimize_benchmark.py
```

**CPU:** практичен только PointNet (74 мс FP32, 56 мс ONNX). Граф-модели — O(N²) на CPU.

```bash
# ONNX-экспорт (opset 18, кроме PointNet++ — dynamic shapes)
python scripts/export_netron_models.py
```

---

## Система экспериментов

### MLflow

Все метрики пишутся в `sqlite:///mlflow.db`. Просмотр:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Sweeps (автоматические серии запусков)

`scripts/run_sweep.py` читает YAML-конфиги из `configs/sweeps/`. Поддерживает три режима:

| Режим | Ключи YAML | Описание |
|-------|-----------|----------|
| `runs` | `runs:` | Список независимых запусков (для KD, сравнений) |
| `matrix` | `models:` + `loss_configs:` | Полная матрица модель × loss |
| single-axis | `batch_sizes:` / `class_balance_betas:` | Вариация одного параметра |

Все режимы поддерживают `--resume` (авто-возобновление прерванного) и `--only_model`.

**Проведённые sweeps:**

| Sweep | Конфиг | Результат |
|-------|--------|-----------|
| batch_size | `*_batch_sweep.yaml` | Лучшие bs для каждой модели |
| class_balance_beta | `*_beta_sweep.yaml` | β=0.99999 оптимален |
| loss functions | `loss_sweep_all_models.yaml` | Lovász/CE + CB-Loss лучшие |
| comparison Mar18/19 | `comparison_mar18/19.yaml` | Кросс-датасет сравнение |
| **distillation (KD)** | `kd_ldgcnn_flash_small.yaml` | **В процессе** |

Лучшие гиперпараметры по результатам всех sweeps зафиксированы в `configs/best_params.yaml`.

---

## Результаты

### Точность (Mar16, scene-level voting)

| Модель | mIoU | OA |
|--------|------|-----|
| PointTransformer | **61.65%** | — |
| LDGCNNFlash (Full) | 60.88% | — |
| LDGCNN-GATv2 | 55.44% | — |
| LDGCNN-LocalWindow | 55.05% | — |
| LDGCNN | 54.76% | — |
| DGCNN | 53.13% | — |
| PointNet++ | 49.15% | — |
| PointNet | 42.47% | — |

Подробные результаты по классам, скорости и оптимизации: `results/vkr_tables.xlsx` (генерация: `python scripts/make_vkr_tables.py`).

### Чувствительность к N (GPU, compile_fp16)

При уменьшении N с 4096 до 1024 LDGCNNFlash ускоряется в ~3× при потере mIoU ~3–5%. PointTransformer при N < 2048 деградирует из-за иерархического FPS.

### Чувствительность к k (GPU)

Оптимальный k=16–20 для граф-моделей. Уменьшение до k=8 даёт ~30% ускорение при потере ~1–2% mIoU.

---

## Известные ограничения

- **PointNet++ AMP:** отключён (`amp: false`). NaN во время обучения из-за нестабильности FPS.
- **Validation mIoU при `--cache_chunked`:** block-level метрика (без scene-level voting). Сравнение моделей корректно (все оценивались одинаково), но абсолютные цифры отличаются от финального `test.py`.
- **PointNet++ ONNX:** не поддерживается (dynamic shapes в `index_put_`).
- **INT8 квантизация:** не даёт ускорения для граф-моделей (узкое место — matmul pairwise distance, а не Conv2d).

---

## Литература

1. **PointNet:** Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. *CVPR 2017*. https://arxiv.org/abs/1612.00593

2. **PointNet++:** Qi, C. R., Yi, L., Su, H., & Guibas, L. J. (2017). PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space. *NeurIPS 2017*. https://arxiv.org/abs/1706.02413

3. **DGCNN:** Wang, Y., Sun, Y., Liu, Z., Sarma, S. E., Bronstein, M. M., & Solomon, J. M. (2019). Dynamic Graph CNN for Learning on Point Clouds. *ACM Transactions on Graphics, 38*(5). https://arxiv.org/abs/1801.07829

4. **LDGCNN:** Zhang, K., Hao, M., Wang, J., de Silva, C. W., & Fu, C. (2019). Linked Dynamic Graph CNN: Learning on Point Cloud via Linking Hierarchical Features. https://arxiv.org/abs/1904.10014

5. **Point Transformer:** Zhao, H., Jiang, L., Jia, J., Torr, P. H. S., & Koltun, V. (2021). Point Transformer. *ICCV 2021*. https://arxiv.org/abs/2012.09164

6. **Flash Attention:** Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022*. https://arxiv.org/abs/2205.14135

7. **Knowledge Distillation:** Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NeurIPS Workshop 2015*. https://arxiv.org/abs/1503.02531

8. **Lovász-Softmax:** Berman, M., Rannen Triki, A., & Blaschko, M. B. (2018). The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the Intersection-Over-Union Measure in Neural Networks. *CVPR 2018*. https://arxiv.org/abs/1705.08790

9. **Class-Balanced Loss:** Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR 2019*. https://arxiv.org/abs/1901.05555

10. **Hessigheim 3D Dataset:** Kölle, M., Laupheimer, D., Schmohl, S., Haala, N., Rottensteiner, F., Wegner, J. D., & Ledoux, H. (2021). The Hessigheim 3D (H3D) Benchmark on Semantic Segmentation of High-Resolution 3D Point Clouds and Image Data for UAV LiDAR and Image-Based Mapping. *ISPRS Open Journal of Photogrammetry and Remote Sensing, 1*, 100001. https://doi.org/10.1016/j.ophoto.2021.100001

---

## Автор

**Новиченков Юрий Дмитриевич** — ВКР, 2025–2026
