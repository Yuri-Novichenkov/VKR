Обучение PointNet

```bash
python train.py \
    --train_data LiDAR/Mar16_train.txt \
    --val_data LiDAR/Mar16_val.txt \
    --model pointnet \
    --save_dir checkpoints/pointnet
```

Обучение PointNet++

```bash
python train.py \
    --train_data LiDAR/Mar16_train.txt \
    --val_data LiDAR/Mar16_val.txt \
    --model pointnet++ \
    --save_dir checkpoints/pointnetpp
```

Сравнение моделей
```bash
python compare_models.py \
    --pointnet_checkpoint checkpoints/pointnet/best_model.pth \
    --pointnetpp_checkpoint checkpoints/pointnetpp/best_model.pth \
    --test_data LiDAR/Mar16_test.txt \
    --save_dir comparison
```