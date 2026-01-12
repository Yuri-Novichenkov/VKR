import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import json

try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_training_history(checkpoint_dir):
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_miou': [],
        'val_miou': []
    }
    
    last_checkpoint = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(last_checkpoint):
        checkpoint = torch.load(last_checkpoint, map_location='cpu', weights_only=False)
        history['epochs'].append(checkpoint.get('epoch', 0))
    
    best_checkpoint = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint):
        checkpoint = torch.load(best_checkpoint, map_location='cpu', weights_only=False)
        if 'train_metrics' in checkpoint and 'val_metrics' in checkpoint:
            train_metrics = checkpoint['train_metrics']
            val_metrics = checkpoint['val_metrics']
            
            history['train_loss'].append(train_metrics.get('loss', 0))
            history['val_loss'].append(val_metrics.get('loss', 0))
            history['train_accuracy'].append(train_metrics.get('accuracy', 0))
            history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            history['train_miou'].append(train_metrics.get('mean_iou', 0))
            history['val_miou'].append(val_metrics.get('mean_iou', 0))
    
    return history


def plot_training_curves(history, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(history['epochs'], history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0, 0].plot(history['epochs'], history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Эпоха', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Кривая обучения: Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['epochs'], history['train_accuracy'], 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[0, 1].plot(history['epochs'], history['val_accuracy'], 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Эпоха', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Кривая обучения: Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].plot(history['epochs'], history['train_miou'], 'b-o', label='Train mIoU', linewidth=2, markersize=6)
    axes[1, 0].plot(history['epochs'], history['val_miou'], 'r-s', label='Val mIoU', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Эпоха', fontsize=12)
    axes[1, 0].set_ylabel('mIoU', fontsize=12)
    axes[1, 0].set_title('Кривая обучения: Mean IoU', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    metrics = ['Loss', 'Accuracy', 'mIoU']
    train_values = [
        history['train_loss'][-1] if history['train_loss'] else 0,
        history['train_accuracy'][-1] if history['train_accuracy'] else 0,
        history['train_miou'][-1] if history['train_miou'] else 0
    ]
    val_values = [
        history['val_loss'][-1] if history['val_loss'] else 0,
        history['val_accuracy'][-1] if history['val_accuracy'] else 0,
        history['val_miou'][-1] if history['val_miou'] else 0
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x - width/2, train_values, width, label='Train', alpha=0.8)
    axes[1, 1].bar(x + width/2, val_values, width, label='Val', alpha=0.8)
    axes[1, 1].set_xlabel('Метрики', fontsize=12)
    axes[1, 1].set_ylabel('Значение', fontsize=12)
    axes[1, 1].set_title('Финальные метрики', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Графики обучения сохранены: {save_path}')
    plt.close()


def plot_confusion_matrix_from_checkpoint(checkpoint_path, save_dir, num_classes=11):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'val_metrics' in checkpoint and 'confusion_matrix' in checkpoint['val_metrics']:
        cm = checkpoint['val_metrics']['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Class {i}' for i in range(num_classes)],
                   yticklabels=[f'Class {i}' for i in range(num_classes)],
                   cbar_kws={'label': 'Количество точек'})
        plt.xlabel('Предсказанные классы', fontsize=12, fontweight='bold')
        plt.ylabel('Истинные классы', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix (Валидация)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix сохранена: {save_path}')
        plt.close()
    else:
        print('Confusion matrix не найдена')


def plot_per_class_iou(checkpoint_path, save_dir, num_classes=11):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'val_metrics' in checkpoint and 'per_class_iou' in checkpoint['val_metrics']:
        ious = checkpoint['val_metrics']['per_class_iou']
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(ious)))
        bars = plt.bar(range(len(ious)), ious, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        for i, (bar, iou) in enumerate(zip(bars, ious)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{iou:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Класс', fontsize=12, fontweight='bold')
        plt.ylabel('IoU', fontsize=12, fontweight='bold')
        plt.title('IoU по классам (Валидация)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(ious)), [f'Class {i}' for i in range(len(ious))])
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim([0, 1.1])
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'per_class_iou.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'График IoU по классам сохранен: {save_path}')
        plt.close()


def visualize_point_cloud_sample(data_file, predictions_file=None, num_points=10000, save_dir='.'):
    data = pd.read_csv(data_file, sep='\t', nrows=num_points)
    
    x = data['X'].values
    y = data['Y'].values
    z = data['Z'].values
    
    # Нормализация координат для лучшей визуализации
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    z = (z - z.mean()) / z.std()
    
    fig = plt.figure(figsize=(16, 6))
    
    if predictions_file and os.path.exists(predictions_file):
        pred_data = pd.read_csv(predictions_file, sep='\t', nrows=num_points)
        if 'Predicted_Classification' in pred_data.columns:
            colors = pred_data['Predicted_Classification'].values[:num_points]
            title_suffix = ' (с предсказаниями)'
        else:
            colors = z  # Используем высоту как цвет
            title_suffix = ''
    else:
        if 'Classification' in data.columns:
            colors = data['Classification'].values
            title_suffix = ' (ground truth)'
        else:
            colors = z
            title_suffix = ''
    
    # цветовая карта
    cmap = plt.cm.tab20 if len(np.unique(colors)) <= 20 else plt.cm.viridis
    
    # XY
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(x, y, c=colors, cmap=cmap, s=1, alpha=0.6)
    ax1.set_xlabel('X', fontsize=11)
    ax1.set_ylabel('Y', fontsize=11)
    ax1.set_title(f'Вид сверху (XY){title_suffix}', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='Класс')
    
    # XZ 
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x, z, c=colors, cmap=cmap, s=1, alpha=0.6)
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Z', fontsize=11)
    ax2.set_title(f'Вид сбоку (XZ){title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Класс')
    
    # 3D 
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(x, y, z, c=colors, cmap=cmap, s=1, alpha=0.6)
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Y', fontsize=11)
    ax3.set_zlabel('Z', fontsize=11)
    ax3.set_title(f'3D вид{title_suffix}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'point_cloud_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Визуализация облака точек сохранена: {save_path}')
    plt.close()


def plot_class_distribution(checkpoint_path, predictions_file=None, save_dir='.'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Распределение из валидации
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'val_metrics' in checkpoint and 'confusion_matrix' in checkpoint['val_metrics']:
        cm = checkpoint['val_metrics']['confusion_matrix']
        val_distribution = cm.sum(axis=1)  # Истинное распределение
        val_distribution = val_distribution / val_distribution.sum()  # Нормализация
        
        axes[0].bar(range(len(val_distribution)), val_distribution, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Класс', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Доля точек', fontsize=11, fontweight='bold')
        axes[0].set_title('Распределение классов (Валидация, Ground Truth)', fontsize=12, fontweight='bold')
        axes[0].set_xticks(range(len(val_distribution)))
        axes[0].set_xticklabels([f'Class {i}' for i in range(len(val_distribution))])
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Распределение из предсказаний
    if predictions_file and os.path.exists(predictions_file):
        pred_data = pd.read_csv(predictions_file, sep='\t')
        if 'Predicted_Classification' in pred_data.columns:
            pred_distribution = pred_data['Predicted_Classification'].value_counts().sort_index()
            pred_distribution = pred_distribution / pred_distribution.sum()
            
            axes[1].bar(pred_distribution.index, pred_distribution.values, 
                       alpha=0.7, color='coral', edgecolor='black')
            axes[1].set_xlabel('Класс', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Доля точек', fontsize=11, fontweight='bold')
            axes[1].set_title('Распределение классов (Тест, Предсказания)', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'График распределения классов сохранен: {save_path}')
    plt.close()


def create_summary_report(checkpoint_path, save_dir):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    report = []
    report.append("="*60)
    report.append("ОТЧЕТ")
    report.append("="*60)
    report.append("")
    
    report.append(f"Эпоха лучшей модели: {checkpoint.get('epoch', 'N/A')}")
    report.append(f"Количество классов: {checkpoint.get('num_classes', 'N/A')}")
    report.append(f"Количество признаков: {checkpoint.get('num_features', 'N/A')}")
    report.append("")
    
    if 'train_metrics' in checkpoint:
        report.append("МЕТРИКИ ОБУЧЕНИЯ:")
        report.append("-"*60)
        train_metrics = checkpoint['train_metrics']
        report.append(f"  Loss: {train_metrics.get('loss', 'N/A'):.4f}")
        report.append(f"  Accuracy: {train_metrics.get('accuracy', 'N/A'):.4f}")
        report.append(f"  mIoU: {train_metrics.get('mean_iou', 'N/A'):.4f}")
        report.append("")
        
        if 'per_class_iou' in train_metrics:
            report.append("  IoU по классам (обучение):")
            for i, iou in enumerate(train_metrics['per_class_iou']):
                report.append(f"    Класс {i}: {iou:.4f}")
        report.append("")
    
    if 'val_metrics' in checkpoint:
        report.append("МЕТРИКИ ВАЛИДАЦИИ:")
        report.append("-"*60)
        val_metrics = checkpoint['val_metrics']
        report.append(f"  Loss: {val_metrics.get('loss', 'N/A'):.4f}")
        report.append(f"  Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
        report.append(f"  mIoU: {val_metrics.get('mean_iou', 'N/A'):.4f}")
        report.append("")
        
        if 'per_class_iou' in val_metrics:
            report.append("  IoU по классам (валидация):")
            for i, iou in enumerate(val_metrics['per_class_iou']):
                report.append(f"    Класс {i}: {iou:.4f}")
    
    report.append("")
    report.append("="*60)
    
    report_text = "\n".join(report)
    
    save_path = os.path.join(save_dir, 'training_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f'Отчет сохранен: {save_path}')
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='Визуализация результатов обучения')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Путь к чекпоинту модели')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Директория с чекпоинтами')
    parser.add_argument('--test_data', type=str, default=None,
                       help='Путь к тестовым данным для визуализации')
    parser.add_argument('--predictions', type=str, default=None,
                       help='Путь к файлу с предсказаниями')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Директория для сохранения визуализаций')
    parser.add_argument('--num_classes', type=int, default=11,
                       help='Количество классов')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Создание визуализаций...")
    print("="*60)
    
    # 1. Графики обучения
    print("\n1. Построение графиков обучения...")
    history = load_training_history(args.checkpoint_dir)
    if history['epochs']:
        plot_training_curves(history, args.output_dir)
    
    # 2. Confusion matrix
    print("\n2. Построение confusion matrix...")
    plot_confusion_matrix_from_checkpoint(args.checkpoint, args.output_dir, args.num_classes)
    
    # 3. IoU по классам
    print("\n3. Построение графика IoU по классам...")
    plot_per_class_iou(args.checkpoint, args.output_dir, args.num_classes)
    
    # 4. Распределение классов
    print("\n4. Построение графика распределения классов...")
    plot_class_distribution(args.checkpoint, args.predictions, args.output_dir)
    
    # 5. Визуализация облака точек
    if args.test_data:
        print("\n5. Визуализация облака точек...")
        visualize_point_cloud_sample(args.test_data, args.predictions, 
                                    num_points=1_400_000, save_dir=args.output_dir)
    
    # 6. Создание отчета
    print("\n6. Создание текстового отчета...")
    create_summary_report(args.checkpoint, args.output_dir)
    
    print("\n" + "="*60)
    print("Все визуализации созданы успешно!")
    print(f"Результаты сохранены в директории: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

