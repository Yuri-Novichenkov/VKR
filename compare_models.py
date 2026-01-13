import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.pointnet import PointNetSegmentation
from models.pointnet_plusplus import PointNetPlusPlusSegmentation
from data.dataset import LiDARDataset
from train import calculate_metrics, validate


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    num_classes = checkpoint['num_classes']
    num_features = checkpoint['num_features']
    model_type = checkpoint.get('model_type', 'pointnet')
    
    if model_type == 'pointnet':
        model = PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    elif model_type == 'pointnet++':
        model = PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
    else:
        raise ValueError(f'Неизвестный тип модели: {model_type}')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, model_type, checkpoint


def compare_models(pointnet_checkpoint, pointnetpp_checkpoint, test_data, 
                   num_points=4096, batch_size=8, save_dir='comparison'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Использование устройства: {device}')
    
    os.makedirs(save_dir, exist_ok=True)
    
    print('\nЗагрузка моделей')
    model_pn, type_pn, checkpoint_pn = load_model(pointnet_checkpoint, device)
    model_pnpp, type_pnpp, checkpoint_pnpp = load_model(pointnetpp_checkpoint, device)
    
    print(f'PointNet загружен: эпоха {checkpoint_pn.get("epoch", "N/A")}, '
          f'mIoU: {checkpoint_pn.get("best_val_iou", "N/A"):.4f}')
    print(f'PointNet++ загружен: эпоха {checkpoint_pnpp.get("epoch", "N/A")}, '
          f'mIoU: {checkpoint_pnpp.get("best_val_iou", "N/A"):.4f}')

    print('\nЗагрузка тестовых данных')
    import pandas as pd
    test_data_sample = pd.read_csv(test_data, sep='\t', nrows=1)
    has_labels = 'Classification' in test_data_sample.columns
    
    test_dataset = LiDARDataset(
        test_data,
        num_points=num_points,
        augment=False,
        has_labels=has_labels
    )
    
    if not has_labels:
        test_dataset.num_classes = checkpoint_pn['num_classes']
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print('\nТестирование PointNet')
    metrics_pn = validate(model_pn, test_loader, device, 
                         checkpoint_pn['num_classes'], 'pointnet')
    
    print('\nТестирование PointNet++')
    metrics_pnpp = validate(model_pnpp, test_loader, device, 
                           checkpoint_pnpp['num_classes'], 'pointnet++')
    
    print('\n' + '='*60)
    
    comparison_data = {
        'Метрика': ['Loss', 'Accuracy', 'mIoU'],
        'PointNet': [
            metrics_pn['loss'],
            metrics_pn['accuracy'],
            metrics_pn['mean_iou']
        ],
        'PointNet++': [
            metrics_pnpp['loss'],
            metrics_pnpp['accuracy'],
            metrics_pnpp['mean_iou']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_names = ['Loss', 'Accuracy', 'mIoU']
    pn_values = [metrics_pn['loss'], metrics_pn['accuracy'], metrics_pn['mean_iou']]
    pnpp_values = [metrics_pnpp['loss'], metrics_pnpp['accuracy'], metrics_pnpp['mean_iou']]
    
    x = range(len(metrics_names))
    width = 0.35
    
    for i, (ax, metric_name, pn_val, pnpp_val) in enumerate(zip(axes, metrics_names, pn_values, pnpp_values)):
        bars1 = ax.bar(i - width/2, pn_val, width, label='PointNet', alpha=0.8, color='steelblue')
        bars2 = ax.bar(i + width/2, pnpp_val, width, label='PointNet++', alpha=0.8, color='coral')
        
        ax.set_ylabel('Значение', fontsize=11)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_xticks([i])
        ax.set_xticklabels([metric_name])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        ax.text(i - width/2, pn_val, f'{pn_val:.4f}', 
               ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, pnpp_val, f'{pnpp_val:.4f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'\nГрафик сравнения сохранен: {save_path}')
    plt.close()

    if 'per_class_iou' in metrics_pn and 'per_class_iou' in metrics_pnpp:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = range(len(metrics_pn['per_class_iou']))
        x = range(len(classes))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], metrics_pn['per_class_iou'], 
              width, label='PointNet', alpha=0.8, color='steelblue')
        ax.bar([i + width/2 for i in x], metrics_pnpp['per_class_iou'], 
              width, label='PointNet++', alpha=0.8, color='coral')
        
        ax.set_xlabel('Класс', fontsize=11, fontweight='bold')
        ax.set_ylabel('IoU', fontsize=11, fontweight='bold')
        ax.set_title('Сравнение IoU по классам', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in classes])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'per_class_iou_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'График IoU по классам сохранен: {save_path}')
        plt.close()

    results_path = os.path.join(save_dir, 'comparison_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(df.to_string(index=False))
        f.write('\n\n')
        f.write('Детальные метрики:\n')
        f.write('-'*60 + '\n')
        f.write(f'\nPointNet:\n')
        f.write(f'  Loss: {metrics_pn["loss"]:.4f}\n')
        f.write(f'  Accuracy: {metrics_pn["accuracy"]:.4f}\n')
        f.write(f'  mIoU: {metrics_pn["mean_iou"]:.4f}\n')
        f.write(f'\nPointNet++:\n')
        f.write(f'  Loss: {metrics_pnpp["loss"]:.4f}\n')
        f.write(f'  Accuracy: {metrics_pnpp["accuracy"]:.4f}\n')
        f.write(f'  mIoU: {metrics_pnpp["mean_iou"]:.4f}\n')
    
    print(f'\nРезультаты сохранены: {results_path}')
    print('\nСравнение завершено')


def main():
    parser = argparse.ArgumentParser(description='Сравнение моделей PointNet и PointNet++')
    parser.add_argument('--pointnet_checkpoint', type=str, required=True,
                       help='Путь к чекпоинту PointNet')
    parser.add_argument('--pointnetpp_checkpoint', type=str, required=True,
                       help='Путь к чекпоинту PointNet++')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Путь к тестовым данным')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Количество точек в облаке')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    parser.add_argument('--save_dir', type=str, default='comparison',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    compare_models(
        args.pointnet_checkpoint,
        args.pointnetpp_checkpoint,
        args.test_data,
        args.num_points,
        args.batch_size,
        args.save_dir
    )


if __name__ == '__main__':
    main()

