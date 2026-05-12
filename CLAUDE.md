# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Project: LiDAR 3D Semantic Segmentation (ВКР)

**Dataset:** Hessigheim 3D (H3D) — UAV LiDAR, 11 classes, ~1.4M points per split.
Files: `Files/Mar16/LiDAR/`, `Files/Mar18/LiDAR/`, `Files/Mar19/LiDAR/`.
Format priority when auto-detecting: `.laz → .las → .txt`.
GroundTruth test file has labels (`Mar16_test_GroundTruth.laz`); plain test file does not.

**MLflow:** All metrics stored in `sqlite:///mlflow.db`. UI:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### Key Commands

```bash
# Train (GPU server)
python scripts/train.py --model ldgcnn_flash --task segmentation --dataset Mar16 \
  --num_points 4096 --batch_size 4 --epochs 100 --lr 0.001 --lr_scheduler cosine --amp \
  --loss_type lovasz --class_balance effective --class_balance_beta 0.99999 --seed 42

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

# Sweep (comparison across models/datasets)
python scripts/run_sweep.py --config configs/sweeps/loss_sweep_all_models.yaml
python scripts/run_sweep.py --config configs/sweeps/comparison_mar18.yaml --resume --only_model "LDGCNNFlash"
```

### Architecture Overview

**6 models**, all in `src/models/`, instantiated via `build_model()` in `src/models/factory.py`:

| model_type | Class | Notes |
|---|---|---|
| `pointnet` | `PointNetSegmentation` | TNet + shared MLP |
| `pointnet++` | `PointNetPlusPlusSegmentation` | FPS + Set Abstraction + FP; **no AMP** (NaN in square_distance) |
| `dgcnn` | `DGCNNSegmentation` | Dynamic graph EdgeConv, kNN in feature space |
| `ldgcnn` | `LDGCNNSegmentation` | EdgeConv in XYZ-space + optional GATv2/local_window attention |
| `pointtransformer` | `PointTransformerSegmentation` | U-Net, vector self-attention; huge compile speedup (11.8×) |
| `ldgcnn_flash` | `LDGCNNFlashSegmentation` | Custom (ВКР): LDGCNN + Flash Self-Attention (SDPA) |

All models share the same `get_loss(predictions, targets, class_weights)` interface.

**Data pipeline** (`src/data/dataset.py` → `LiDARDataset`):
- Reads `.txt`/`.las`/`.laz` via `src/data/io.py`
- Splits scene into overlapping windows of `num_points` points
- Caches point-cloud arrays to disk (`cache_dir`, modes: `off`/`read`/`write`, chunked or monolithic)
- At test/prediction time, `get_cloud_point_indices(sample_idx)` returns original point indices for **voting**: each point collects votes from all windows it appears in, final label = argmax

**Voting** is critical for correct segmentation metrics — without it, boundary points are counted multiple times. `test.py` and `predictions.py` both use voting via `vote_counts` accumulation.

**Loss options** (`train.py --loss_type`): `ce`, `focal`, `lovasz`. Combined with `--class_balance effective --class_balance_beta 0.99999` (CB-Loss). Best config per model in `configs/best_params.yaml`.

**Sweep system** (`scripts/run_sweep.py`):
- Reads YAML configs from `configs/sweeps/`
- `--resume`: skips runs where `metrics.json` already exists in output dir
- `--only_model "<display_name>"`: runs only one model (for 24-hour server limit)
- Matrix mode writes to `output_root:` from YAML (comparison sweeps → `experiments/comparison/`)

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

### Inference Optimization Results (Mar16, GPU RTX A5000, batch=1)

Best mode per model:
- **PointNet**: `compile_fp16` → 1.38 ms (1.8× vs fp32)
- **DGCNN**: `compile_fp16` → 3.7 ms (3.9×)
- **LDGCNN**: `compile_fp16` → 17.1 ms (2.2×)
- **PointTransformer**: `compile_fp32` → 12.8 ms (**11.8×** — inductor unrolls attention loop)
- **LDGCNNFlash**: `compile_fp16` → 5.9 ms (2.5×)
- **PointNet++**: no mode helps (~178 ms, FPS is sequential CPU-bound)

CPU: only PointNet practical (74 ms fp32, 56 ms ONNX). Graph-based models are O(N²) on CPU.
PointNet++ ONNX export fails (dynamic shapes in `index_put_`). Use opset 18 for others.

### Class Legend

Hessigheim 11 classes defined in `configs/classes/hessigheim.yaml`:
`0=Low Vegetation, 1=Impervious Surface, 2=Vehicle, 3=Urban Furniture, 4=Roof, 5=Facade, 6=Shrub, 7=Tree, 8=Soil/Gravel, 9=Vertical Surface, 10=Chimney`
