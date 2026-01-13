import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import argparse
from pathlib import Path

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")


def create_results_summary(pointnet_path=None, pointnetpp_path=None, output_dir='presentation'):
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    # PointNet результаты
    if pointnet_path and os.path.exists(pointnet_path):
        pn_checkpoint = torch.load(pointnet_path, map_location='cpu', weights_only=False)
        if 'val_metrics' in pn_checkpoint:
            val = pn_checkpoint['val_metrics']
            results.append({
                'Модель': 'PointNet',
                'mIoU (%)': f"{val.get('mean_iou', 0)*100:.2f}",
                'Accuracy (%)': f"{val.get('accuracy', 0)*100:.2f}",
                'Loss': f"{val.get('loss', 0):.4f}",
                'Параметров': f"{sum(p.numel() for p in pn_checkpoint['model_state_dict'].values()):,}",
                'Эпоха': pn_checkpoint.get('epoch', 'N/A')
            })
    
    # PointNet++ результаты
    if pointnetpp_path and os.path.exists(pointnetpp_path):
        pnpp_checkpoint = torch.load(pointnetpp_path, map_location='cpu', weights_only=False)
        if 'val_metrics' in pnpp_checkpoint:
            val = pnpp_checkpoint['val_metrics']
            results.append({
                'Модель': 'PointNet++',
                'mIoU (%)': f"{val.get('mean_iou', 0)*100:.2f}",
                'Accuracy (%)': f"{val.get('accuracy', 0)*100:.2f}",
                'Loss': f"{val.get('loss', 0):.4f}",
                'Параметров': f"{sum(p.numel() for p in pnpp_checkpoint['model_state_dict'].values()):,}",
                'Эпоха': pnpp_checkpoint.get('epoch', 'N/A')
            })
    
    if results:
        df = pd.DataFrame(results)
        
        csv_path = os.path.join(output_dir, 'results_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Таблица результатов сохранена: {csv_path}")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Сравнение результатов моделей', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(output_dir, 'results_table.png'), dpi=300, bbox_inches='tight')
        print(f"График таблицы сохранен: {os.path.join(output_dir, 'results_table.png')}")
        plt.close()
        
        return df
    else:
        print("Результаты не найдены")
        return None


def create_metrics_comparison_chart(pointnet_path=None, pointnetpp_path=None, output_dir='presentation'):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_data = {'Модель': [], 'mIoU': [], 'Accuracy': []}
    
    if pointnet_path and os.path.exists(pointnet_path):
        checkpoint = torch.load(pointnet_path, map_location='cpu', weights_only=False)
        if 'val_metrics' in checkpoint:
            val = checkpoint['val_metrics']
            metrics_data['Модель'].append('PointNet')
            metrics_data['mIoU'].append(val.get('mean_iou', 0))
            metrics_data['Accuracy'].append(val.get('accuracy', 0))
    
    if pointnetpp_path and os.path.exists(pointnetpp_path):
        checkpoint = torch.load(pointnetpp_path, map_location='cpu', weights_only=False)
        if 'val_metrics' in checkpoint:
            val = checkpoint['val_metrics']
            metrics_data['Модель'].append('PointNet++')
            metrics_data['mIoU'].append(val.get('mean_iou', 0))
            metrics_data['Accuracy'].append(val.get('accuracy', 0))
    
    if not metrics_data['Модель']:
        print("Нет данных для сравнения")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    models = metrics_data['Модель']
    mious = metrics_data['mIoU']
    accs = metrics_data['Accuracy']
    
    bars1 = ax1.bar(models, mious, color=['steelblue', 'coral'][:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('mIoU', fontsize=13, fontweight='bold')
    ax1.set_title('Сравнение mIoU', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, max(mious) * 1.2 if mious else 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, mious):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}\n({val*100:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    bars2 = ax2.bar(models, accs, color=['steelblue', 'coral'][:len(models)], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax2.set_title('Сравнение Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, accs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}\n({val*100:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сравнения метрик сохранен: {save_path}")
    plt.close()


def create_final_report(pointnetpp_path, output_dir='presentation'):
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = torch.load(pointnetpp_path, map_location='cpu', weights_only=False)
    
    report = []
    
    if 'val_metrics' in checkpoint:
        val = checkpoint['val_metrics']
        report.append("РЕЗУЛЬТАТЫ POINTNET++:")
        report.append("-"*70)
        report.append(f"Валидационный mIoU: {val.get('mean_iou', 0)*100:.2f}%")
        report.append(f"Accuracy на валидации: {val.get('accuracy', 0)*100:.2f}%")
        report.append(f"Loss: {val.get('loss', 0):.4f}")
        report.append(f"Эпоха лучшей модели: {checkpoint.get('epoch', 'N/A')}")
        report.append("")
        
        if 'per_class_iou' in val:
            report.append("IoU по классам (валидация):")
            for i, iou in enumerate(val['per_class_iou']):
                report.append(f"  Класс {i}: {iou*100:.2f}%")
            report.append("")
    
    report_text = "\n".join(report)
    
    save_path = os.path.join(output_dir, 'final_report.txt')
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"Финальный отчет сохранен: {save_path}")
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pointnet', type=str, default=None,
                       help='Путь к чекпоинту PointNet')
    parser.add_argument('--pointnetpp', type=str, default='checkpoints/best_model.pth',
                       help='Путь к чекпоинту PointNet++')
    parser.add_argument('--output_dir', type=str, default='presentation',
                       help='Директория для сохранения материалов')
    
    args = parser.parse_args()
    
    print("\n1. Создание итоговой таблицы результатов")
    create_results_summary(args.pointnet, args.pointnetpp, args.output_dir)

    print("\n2. Создание графика сравнения метрик")
    create_metrics_comparison_chart(args.pointnet, args.pointnetpp, args.output_dir)
    
    # 3. Финальный отчет
    print("\n3. Создание финального отчета")
    create_final_report(args.pointnetpp, args.output_dir)
    
    print("\n" + "="*70)
    print(f"Все файлы сохранены в директории: {args.output_dir}")


if __name__ == '__main__':
    main()

