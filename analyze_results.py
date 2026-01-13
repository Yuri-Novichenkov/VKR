import torch
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def load_checkpoint_metrics(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    metrics = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'model_type': checkpoint.get('model_type', 'unknown'),
        'num_params': sum(p.numel() for p in checkpoint['model_state_dict'].values()) if 'model_state_dict' in checkpoint else 0
    }
    
    if 'train_metrics' in checkpoint:
        train = checkpoint['train_metrics']
        metrics['train_loss'] = train.get('loss', 0)
        metrics['train_accuracy'] = train.get('accuracy', 0)
        metrics['train_miou'] = train.get('mean_iou', 0)
        metrics['train_per_class_iou'] = train.get('per_class_iou', [])
    
    if 'val_metrics' in checkpoint:
        val = checkpoint['val_metrics']
        metrics['val_loss'] = val.get('loss', 0)
        metrics['val_accuracy'] = val.get('accuracy', 0)
        metrics['val_miou'] = val.get('mean_iou', 0)
        metrics['val_per_class_iou'] = val.get('per_class_iou', [])
    
    return metrics


def create_comparison_table(pointnet_path, pointnetpp_path, save_path):
    """Создание таблицы сравнения моделей"""
    pn_metrics = load_checkpoint_metrics(pointnet_path)
    pnpp_metrics = load_checkpoint_metrics(pointnetpp_path)
    
    comparison = {
        'Метрика': [
            'Эпоха',
            'Параметров',
            'Train Loss',
            'Train Accuracy',
            'Train mIoU',
            'Val Loss',
            'Val Accuracy',
            'Val mIoU'
        ],
        'PointNet': [
            pn_metrics.get('epoch', 'N/A'),
            f"{pn_metrics.get('num_params', 0):,}",
            f"{pn_metrics.get('train_loss', 0):.4f}",
            f"{pn_metrics.get('train_accuracy', 0):.4f}",
            f"{pn_metrics.get('train_miou', 0):.4f}",
            f"{pn_metrics.get('val_loss', 0):.4f}",
            f"{pn_metrics.get('val_accuracy', 0):.4f}",
            f"{pn_metrics.get('val_miou', 0):.4f}"
        ],
        'PointNet++': [
            pnpp_metrics.get('epoch', 'N/A'),
            f"{pnpp_metrics.get('num_params', 0):,}",
            f"{pnpp_metrics.get('train_loss', 0):.4f}",
            f"{pnpp_metrics.get('train_accuracy', 0):.4f}",
            f"{pnpp_metrics.get('train_miou', 0):.4f}",
            f"{pnpp_metrics.get('val_loss', 0):.4f}",
            f"{pnpp_metrics.get('val_accuracy', 0):.4f}",
            f"{pnpp_metrics.get('val_miou', 0):.4f}"
        ]
    }
    
    df = pd.DataFrame(comparison)
    
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f'\nТаблица сравнения сохранена: {save_path}')
    print('\n' + df.to_string(index=False))
    
    return df


def plot_comparison_metrics(pointnet_path, pointnetpp_path, save_dir):
    pn_metrics = load_checkpoint_metrics(pointnet_path)
    pnpp_metrics = load_checkpoint_metrics(pointnetpp_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].bar(['PointNet', 'PointNet++'], 
                   [pn_metrics.get('val_loss', 0), pnpp_metrics.get('val_loss', 0)],
                   color=['steelblue', 'coral'], alpha=0.8)
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Сравнение Loss (Валидация)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy
    axes[0, 1].bar(['PointNet', 'PointNet++'],
                   [pn_metrics.get('val_accuracy', 0), pnpp_metrics.get('val_accuracy', 0)],
                   color=['steelblue', 'coral'], alpha=0.8)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Сравнение Accuracy (Валидация)', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # mIoU
    axes[1, 0].bar(['PointNet', 'PointNet++'],
                   [pn_metrics.get('val_miou', 0), pnpp_metrics.get('val_miou', 0)],
                   color=['steelblue', 'coral'], alpha=0.8)
    axes[1, 0].set_ylabel('mIoU', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Сравнение mIoU (Валидация)', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Параметры
    axes[1, 1].bar(['PointNet', 'PointNet++'],
                   [pn_metrics.get('num_params', 0) / 1e6, pnpp_metrics.get('num_params', 0) / 1e6],
                   color=['steelblue', 'coral'], alpha=0.8)
    axes[1, 1].set_ylabel('Параметров (млн)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Сравнение размера моделей', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for ax, values in zip([axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]], 
                         [[pn_metrics.get('val_loss', 0), pnpp_metrics.get('val_loss', 0)],
                          [pn_metrics.get('val_accuracy', 0), pnpp_metrics.get('val_accuracy', 0)],
                          [pn_metrics.get('val_miou', 0), pnpp_metrics.get('val_miou', 0)],
                          [pn_metrics.get('num_params', 0) / 1e6, pnpp_metrics.get('num_params', 0) / 1e6]]):
        for i, (bar, val) in enumerate(zip(ax.patches, values)):
            height = bar.get_height()
            if i == 0:  # PointNet
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}' if val < 1 else f'{val:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            else:  # PointNet++
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}' if val < 1 else f'{val:.2f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'models_comparison_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'График сравнения метрик сохранен: {save_path}')
    plt.close()


def plot_per_class_comparison(pointnet_path, pointnetpp_path, save_dir):
    """Сравнение IoU по классам"""
    pn_metrics = load_checkpoint_metrics(pointnet_path)
    pnpp_metrics = load_checkpoint_metrics(pointnetpp_path)
    
    pn_ious = pn_metrics.get('val_per_class_iou', [])
    pnpp_ious = pnpp_metrics.get('val_per_class_iou', [])
    
    if not pn_ious or not pnpp_ious:
        print('IoU по классам не найдены в чекпоинтах')
        return
    
    num_classes = max(len(pn_ious), len(pnpp_ious))
    classes = range(num_classes)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = range(num_classes)
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], pn_ious[:num_classes], 
                   width, label='PointNet', alpha=0.8, color='steelblue', edgecolor='black')
    bars2 = ax.bar([i + width/2 for i in x], pnpp_ious[:num_classes], 
                   width, label='PointNet++', alpha=0.8, color='coral', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Класс', fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение IoU по классам (Валидация)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in classes])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'per_class_iou_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'График сравнения IoU по классам сохранен: {save_path}')
    plt.close()


def create_summary_report(pointnet_path, pointnetpp_path, save_path):
    pn_metrics = load_checkpoint_metrics(pointnet_path)
    pnpp_metrics = load_checkpoint_metrics(pointnetpp_path)
    
    report = []
    report.append("="*70)
    report.append("СРАВНЕНИЕ МОДЕЛЕЙ POINTNET И POINTNET++")
    report.append("="*70)
    report.append("")
    
    report.append("ОБЩИЕ МЕТРИКИ:")
    report.append("-"*70)
    report.append(f"{'Метрика':<25} {'PointNet':<20} {'PointNet++':<20}")
    report.append("-"*70)
    report.append(f"{'Эпоха':<25} {pn_metrics.get('epoch', 'N/A'):<20} {pnpp_metrics.get('epoch', 'N/A'):<20}")
    report.append(f"{'Параметров':<25} {pn_metrics.get('num_params', 0):,} {pnpp_metrics.get('num_params', 0):,}")
    report.append("")
    
    report.append("МЕТРИКИ ОБУЧЕНИЯ:")
    report.append("-"*70)
    report.append(f"{'Метрика':<25} {'PointNet':<20} {'PointNet++':<20}")
    report.append("-"*70)
    report.append(f"{'Loss':<25} {pn_metrics.get('train_loss', 0):.4f} {pnpp_metrics.get('train_loss', 0):.4f}")
    report.append(f"{'Accuracy':<25} {pn_metrics.get('train_accuracy', 0):.4f} {pnpp_metrics.get('train_accuracy', 0):.4f}")
    report.append(f"{'mIoU':<25} {pn_metrics.get('train_miou', 0):.4f} {pnpp_metrics.get('train_miou', 0):.4f}")
    report.append("")
    
    report.append("МЕТРИКИ ВАЛИДАЦИИ:")
    report.append("-"*70)
    report.append(f"{'Метрика':<25} {'PointNet':<20} {'PointNet++':<20}")
    report.append("-"*70)
    report.append(f"{'Loss':<25} {pn_metrics.get('val_loss', 0):.4f} {pnpp_metrics.get('val_loss', 0):.4f}")
    report.append(f"{'Accuracy':<25} {pn_metrics.get('val_accuracy', 0):.4f} {pnpp_metrics.get('val_accuracy', 0):.4f}")
    report.append(f"{'mIoU':<25} {pn_metrics.get('val_miou', 0):.4f} {pnpp_metrics.get('val_miou', 0):.4f}")
    report.append("")
    
    # IoU по классам
    pn_ious = pn_metrics.get('val_per_class_iou', [])
    pnpp_ious = pnpp_metrics.get('val_per_class_iou', [])
    
    if pn_ious and pnpp_ious:
        report.append("IoU ПО КЛАССАМ (Валидация):")
        report.append("-"*70)
        report.append(f"{'Класс':<10} {'PointNet':<15} {'PointNet++':<15} {'Разница':<15}")
        report.append("-"*70)
        for i in range(max(len(pn_ious), len(pnpp_ious))):
            pn_iou = pn_ious[i] if i < len(pn_ious) else 0
            pnpp_iou = pnpp_ious[i] if i < len(pnpp_ious) else 0
            diff = pnpp_iou - pn_iou
            report.append(f"{'Class ' + str(i):<10} {pn_iou:<15.4f} {pnpp_iou:<15.4f} {diff:+.4f}")
        report.append("")
    
    report.append("")
    report.append("="*70)
    
    report_text = "\n".join(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f'\n {save_path}')
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pointnet_checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Путь к чекпоинту PointNet')
    parser.add_argument('--pointnetpp_checkpoint', type=str, default='checkpoints/pointnetpp/best_model.pth',
                       help='Путь к чекпоинту PointNet++')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Директория для сохранения результатов')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)

    print("\n1. Создание таблицы сравнения")
    create_comparison_table(
        args.pointnet_checkpoint,
        args.pointnetpp_checkpoint,
        os.path.join(args.output_dir, 'comparison_table.csv')
    )
    
    print("\n2. Создание графиков сравнения метрик")
    plot_comparison_metrics(
        args.pointnet_checkpoint,
        args.pointnetpp_checkpoint,
        args.output_dir
    )
    
    print("\n3. Создание графика сравнения IoU по классам")
    plot_per_class_comparison(
        args.pointnet_checkpoint,
        args.pointnetpp_checkpoint,
        args.output_dir
    )
    
    print("\n4. Создание отчета")
    create_summary_report(
        args.pointnet_checkpoint,
        args.pointnetpp_checkpoint,
        os.path.join(args.output_dir, 'summary_report.txt')
    )
    
    print("\n" + "="*70)
    print("Анализ завершен")
    print(f"{args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()

