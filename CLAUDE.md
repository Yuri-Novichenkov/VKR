# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: LiDAR 3D Semantic Segmentation (ВКР)

**Dataset:** Hessigheim 3D (H3D) — UAV LiDAR, 11 classes, ~1.4M points per split.
Files: `Files/Mar16/LiDAR/`, `Files/Mar18/LiDAR/`, `Files/Mar19/LiDAR/`.
Format priority when auto-detecting: `.laz → .las → .txt`.
GroundTruth test file has labels (`Mar16_test_GroundTruth.laz`); plain test file does not.

**MLflow:** All metrics stored in `sqlite:///mlflow.db`. UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Setup

```bash
# Install core dependencies
pip install -r requirements.txt

# Install dev/analysis tools (profiling, ONNX export, torch_cluster)
pip install -r requirements-dev.txt

# PyTorch must be installed separately for your CUDA version
# See https://pytorch.org/get-started/locally/
```

### Key Commands

```bash
# Train (GPU server)
python scripts/train.py --model ldgcnn_flash --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 4 --epochs 100 --lr 0.001 --lr_scheduler cosine --amp \
  --loss_type lovasz --class_balance effective --class_balance_beta 0.99999 --seed 42

# LDGCNN with attention variant (attention_type: none | gatv2 | local_window)
python scripts/train.py --model ldgcnn --attention_type gatv2 \
  --attention_k 16 --attention_heads 4 --attention_dropout 0.1 \
  --loss_type ce --class_balance effective --class_balance_beta 0.9999 ...

# Test (GroundTruth file has labels)
python scripts/test.py --checkpoint <path> \
  --dataset Mar16 --num_points 4096 --batch_size 8 --device cuda

# Predictions (CPU laptop, ~1–30 min depending on model)
python scripts/predictions.py --checkpoint <path> \
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz \
  --output_root predictions --device cpu

# Visualization (Open3D interactive + legend PNG saved to figures/legend.png)
python scripts/visualization.py \
  --data Files/Mar16/LiDAR/Mar16_test_GroundTruth.laz \
  --predictions predictions/ldgcnn_flash/Mar16_test_GroundTruth/Mar16_test_GroundTruth_predictions.txt \
  --color_by pred --num_classes 11 \
  --class_config configs/classes/hessigheim.yaml \
  --legend --max_points 300000

# Inference benchmark (GPU)
python scripts/benchmark_inference.py --all_mar16 \
  --modes fp32 fp16 compile_fp32 compile_fp16 --batch_size 1 \
  --output_csv results/benchmark_gpu.csv

# N-point sensitivity analysis (speed + mIoU for N=512–4096, GPU or CPU)
python scripts/n_sensitivity.py  # outputs results/n_sensitivity_{device}.xlsx

# k-neighbor sensitivity analysis (speed + mIoU vs k, GPU only)
python scripts/k_sensitivity.py  # outputs results/k_sensitivity.xlsx

# Generate main VKR report (7-sheet Excel with all metrics, charts, color coding)
python scripts/make_vkr_tables.py  # outputs results/vkr_tables.xlsx

# Knowledge Distillation sweep (3 runs: Small baseline, Full+KD, Small+KD)
python scripts/run_sweep.py --config configs/sweeps/kd_ldgcnn_flash_small.yaml
python scripts/run_sweep.py --config configs/sweeps/kd_ldgcnn_flash_small.yaml --resume --only_model "LDGCNNFlashSmall+KD"

# Обучение вручную: LDGCNNFlash-Small с KD от PointTransformer
python scripts/train.py --model ldgcnn_flash --flash_channels 32 64 128 \
  --task segmentation --dataset Mar16 --num_points 4096 --batch_size 16 \
  --epochs 100 --lr 0.001 --lr_scheduler cosine --amp --seed 42 \
  --loss_type ce --class_balance effective --class_balance_beta 0.99999 \
  --kd_teacher_checkpoint checkpoints/loss_sweep/pointtransformer/segmentation/loss_lovasz_g2p0__cb_effective_b0p99999/mar16/best_model.pth \
  --kd_alpha 0.5 --kd_temperature 4.0

# Sweep (comparison across models/datasets)
python scripts/run_sweep.py --config configs/sweeps/loss_sweep_all_models.yaml
python scripts/run_sweep.py --config configs/sweeps/comparison_mar18.yaml --resume --only_model "LDGCNNFlash"
```

### Architecture Overview

**8 model variants**, all in `src/models/`, instantiated via `build_model()` in `src/models/factory.py`:

| model_type | Variant / Class | Mar16 mIoU | Notes |
|---|---|---|---|
| `pointnet` | `PointNetSegmentation` | 42.47% | TNet + shared MLP |
| `pointnet++` | `PointNetPlusPlusSegmentation` | 49.15% | FPS + Set Abstraction + FP; **no AMP** |
| `dgcnn` | `DGCNNSegmentation` | 53.13% | Dynamic graph EdgeConv, kNN in feature space |
| `ldgcnn` | `LDGCNNSegmentation` (none) | 54.76% | EdgeConv in XYZ-space, no attention |
| `ldgcnn` | `LDGCNNSegmentation` (local_window) | 55.05% | + local window attention |
| `ldgcnn` | `LDGCNNSegmentation` (gatv2) | 55.44% | + GATv2 attention ← best graph model |
| `ldgcnn_flash` | `LDGCNNFlashSegmentation` | 60.88% | LDGCNN + Flash Self-Attention (SDPA) |
| `pointtransformer` | `PointTransformerSegmentation` | 61.65% | U-Net, vector self-attention ← best overall |

LDGCNN attention variant is set via `--attention_type {none,gatv2,local_window}` at train time (stored in checkpoint; `build_model()` reads it automatically).

All models share the same `get_loss(predictions, targets, class_weights)` interface.
`build_model()` is the single source of truth — all scripts use it to prevent train/test divergence.

**Utilities** (`src/utils/`):
- `point_ops.py`: `knn(x, k, fast=False)`, `get_graph_feature(x, k, idx, fast_knn)`, `EdgeConv` — shared graph convolution primitives used by DGCNN, LDGCNN, and LDGCNNFlash
- `metrics.py`: `calculate_metrics(predictions, targets, num_classes, task)` — returns mIoU per class for segmentation, accuracy for classification

**Data pipeline** (`src/data/dataset.py` → `LiDARDataset`):
- Reads `.txt`/`.las`/`.laz` via `src/data/io.py:load_dataframe()`
- Splits scene into overlapping windows of `num_points` points
- Caches point-cloud arrays to disk (`cache_dir`, modes: `off`/`read`/`write`, chunked or monolithic)
- Normalization: z-score computed on train, stored in `self.normalize_stats`, passed to val/test via `normalize_stats=` arg
- At test/prediction time, `get_cloud_point_indices(sample_idx)` returns original point indices for **voting**: each point collects votes from all windows it appears in, final label = argmax

**Voting** is critical for correct segmentation metrics — without it, boundary points are counted multiple times. `test.py` and `predictions.py` both use voting via `vote_counts` accumulation.

**Loss options** (`train.py --loss_type`): `ce`, `focal`, `lovasz`. Combined with `--class_balance effective --class_balance_beta 0.99999` (CB-Loss). Best config per model in `configs/best_params.yaml`.

**Sweep system** (`scripts/run_sweep.py`):
- Reads YAML configs from `configs/sweeps/`
- `--resume`: skips runs where `metrics.json` already exists in output dir
- `--only_model "<display_name>"`: runs only one model (for 24-hour server limit)
- Matrix mode writes to `output_root:` from YAML (comparison sweeps → `experiments/comparison/`)

### Knowledge Distillation (финал ВКР)

Учитель: PointTransformer (61.65% mIoU). Студент: LDGCNNFlash-Small (каналы 32/64/128 вместо 64/128/256).

**Новые аргументы `train.py`:**
- `--flash_channels C1 C2 C3` — каналы для LDGCNNFlash (default `64 128 256`; small: `32 64 128`)
- `--kd_teacher_checkpoint PATH` — чекпоинт учителя (любая модель из `build_model`)
- `--kd_alpha FLOAT` — вес KD loss (0 = только task, 1 = только KD; default `0.5`)
- `--kd_temperature FLOAT` — температура softmax (default `4.0`)

**Loss formula:** `L = (1-α) · L_task + α · T² · KL(student/T ‖ teacher/T)`

Учитель загружается через `load_teacher()` (замораживается `requires_grad=False`), инференс — в FP32 с `torch.no_grad()` до autocast-блока студента.

Три run-а в `configs/sweeps/kd_ldgcnn_flash_small.yaml`:
1. `LDGCNNFlashSmall_baseline` — Small без KD (нижняя граница)
2. `LDGCNNFlash_full_kd` — Full + KD (проверка прироста от учителя)
3. `LDGCNNFlashSmall_kd` — Small + KD (ключевой результат)

`run_sweep.py` теперь поддерживает режим `runs:` в YAML (список независимых запусков с произвольными params, включая list-аргументы типа `flash_channels`).

### Report Generation

`scripts/make_vkr_tables.py` produces `results/vkr_tables.xlsx` — the primary deliverable. It contains 7 sheets:
1. **Сравнение моделей** — 8 models × 14 metrics (params, GMACs, mIoU, training time, GPU/CPU bench)
2. **GPU-оптимизация** — all 4 precision modes (fp32/fp16/compile variants) with timing and VRAM
3. **N-sensitivity** — speed + mIoU for N=512,1024,2048,3072,4096 on CPU and GPU
4. **k-sensitivity** — speed + mIoU vs k for graph models (DGCNN, LDGCNN, LDGCNNFlash, PointTransformer)
5. **INT8 квантизация** — dynamic quantization speedup per model
6. **Оптимальный N** — speed/accuracy tradeoff recommendations
7. **Методы оптимизации** — summary of 8 optimization techniques

The data in this script is **hardcoded** from final experiment runs. To update it, edit the dictionaries at the top of `make_vkr_tables.py` and re-run.

### Critical AMP / FP16 Bugs (already fixed, do not revert)

1. **`square_distance` in `pointnet_plusplus.py` and `point_transformer.py`**: `torch.matmul` runs FP16 under autocast even with `.float()` inputs. Fixed with `torch.amp.autocast(device_type=..., enabled=False)` wrapper.

2. **`farthest_point_sample` in `pointnet_plusplus.py`**: `distance` buffer must match `xyz.dtype` for `index_put_` to work in FP16:
   ```python
   distance = torch.full((B, N), float('inf'), dtype=xyz.dtype, device=device)
   ```

3. **`FeaturePropagation` in `pointnet_plusplus.py`**: IDW weight must be cast to `points2.dtype` before multiply.

4. **`FlashAttentionBlock` in `ldgcnn_flash.py`**: Do NOT call `self.norm1(x.float())` — causes dtype mismatch when `model.half()`. LayerNorm is fp32-preserved under autocast automatically; call `self.norm1(x)` directly.

5. **PointNet++ AMP**: Disable entirely (`amp: false` in training). NaN occurs during training even with above fixes due to FPS instability.

### Known Limitations

- **Validation mIoU при `--cache_chunked`**: В train.py scene-level voting отключается если val_dataset chunked (см. `use_voting` в evaluate()). Все sweep-эксперименты запускались с `--cache_chunked`, поэтому best_val_miou в MLflow — block-level метрика. Финальный test.py использует scene-level voting. **Сравнение моделей между собой корректно** (все оценивались одинаково), но абсолютные цифры могут незначительно отличаться.

- **torch_cluster (fast_knn)**: Для плотных батчей N=4096 torch_cluster оказывается ~14× медленнее matmul-based kNN, потому что cuBLAS GEMM эффективнее разреженного алгоритма. fast_knn=False — дефолт; fast_knn=True нужен только для экспериментов.

- **PointTransformer при N < 2048**: mIoU резко падает из-за иерархического FPS — каждый уровень вдвое уменьшает число точек; при малых N нижние уровни вырождаются.

- **PointNet++ ONNX export**: Завершается с ошибкой из-за динамических форм в `index_put_`. Для остальных моделей использовать opset 18.

- **Dynamic INT8 quantization**: Не даёт ускорения для graph-based моделей (узкое место — matmul в pairwise_distance, а не Conv2d).

### Inference Optimization Results (Mar16, GPU RTX A5000, batch=1)

Best mode per model:
- **PointNet**: `compile_fp16` → 1.38 ms (1.8× vs fp32)
- **DGCNN**: `compile_fp16` → 3.7 ms (3.9×)
- **LDGCNN**: `compile_fp16` → 17.1 ms (2.2×)
- **LDGCNNFlash**: `compile_fp16` → 5.9 ms (2.5×)
- **PointTransformer**: `compile_fp32` → 12.8 ms (**11.8×** — inductor unrolls attention loop)
- **PointNet++**: no mode helps (~178 ms, FPS is sequential CPU-bound)

CPU: only PointNet practical (74 ms fp32, 56 ms ONNX). Graph-based models are O(N²) on CPU.

### Class Legend

Hessigheim 11 classes defined in `configs/classes/hessigheim.yaml`:
`0=Low Vegetation, 1=Impervious Surface, 2=Vehicle, 3=Urban Furniture, 4=Roof, 5=Facade, 6=Shrub, 7=Tree, 8=Soil/Gravel, 9=Vertical Surface, 10=Chimney`
