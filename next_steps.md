Если PointNet++ обучен в отдельной директории:
```bash
python view_results.py --checkpoint checkpoints/pointnetpp/best_model.pth
```

```bash
python view_results.py --checkpoint checkpoints/best_model.pth
```

```bash
python test.py --checkpoint checkpoints/best_model.pth --test_data LiDAR/Mar16_test.txt --save_results results_pointnetpp.txt
```

```bash
python train.py --train_data LiDAR/Mar16_train.txt --val_data LiDAR/Mar16_val.txt --model pointnet --save_dir checkpoints/pointnet

python test.py --checkpoint checkpoints/pointnet/best_model.pth --test_data LiDAR/Mar16_test.txt --save_results results_pointnet.txt
```

```bash
python compare_models.py --pointnet_checkpoint checkpoints/best_model.pth --pointnetpp_checkpoint checkpoints/pointnetpp/best_model.pth --test_data LiDAR/Mar16_test.txt --save_dir comparison
```

Для PointNet:
```bash
python visualize.py --checkpoint checkpoints/best_model.pth --test_data LiDAR/Mar16_test.txt --predictions predictions_pointnet.txt --output_dir visualizations/pointnet
```

Для PointNet++:
```bash
python visualize.py --checkpoint checkpoints/pointnetpp/best_model.pth --test_data LiDAR/Mar16_test.txt --predictions predictions_pointnetpp.txt --output_dir visualizations/pointnetpp
```

PointNet:
```bash
python save_predictions.py --checkpoint checkpoints/best_model.pth --test_data LiDAR/Mar16_test.txt --output predictions_pointnet.txt
```

PointNet++:
```bash
python save_predictions.py --checkpoint checkpoints/pointnetpp/best_model.pth --test_data LiDAR/Mar16_test.txt --output predictions_pointnetpp.txt
```