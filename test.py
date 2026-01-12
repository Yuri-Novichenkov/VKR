"""
Скрипт для тестирования обученной модели PointNet
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from models.pointnet import PointNetSegmentation
from data.dataset import LiDARDataset


def calculate_metrics(predictions, targets, num_classes, class_names=None):
    """
    Вычисление метрик качества
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Точность
    accuracy = accuracy_score(targets.flatten(), predictions.flatten())
    
    # Confusion matrix
    cm = confusion_matrix(targets.flatten(), predictions.flatten(), 
                         labels=list(range(num_classes)))
    
    # IoU для каждого класса
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = cm[i, :].sum() + cm[:, i].sum() - intersection
        if union > 0:
            iou = intersection / union
            ious.append(iou)
        else:
            ious.append(0.0)
    
    mean_iou = np.mean(ious)
    
    # Precision, Recall, F1 для каждого класса
    report = classification_report(
        targets.flatten(), 
        predictions.flatten(),
        labels=list(range(num_classes)),
        target_names=class_names if class_names else [f'Class {i}' for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'per_class_iou': ious,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Визуализация confusion matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Confusion matrix сохранена в {save_path}')
    else:
        plt.show()
    plt.close()


def test_model(model, test_loader, device, num_classes, class_names=None, has_labels=True):
    """
    Тестирование модели
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            predictions, _, _ = model(features)
            
            # Получение вероятностей
            probs = torch.softmax(predictions, dim=2)
            pred_classes = torch.argmax(predictions, dim=2)
            
            all_predictions.append(pred_classes)
            all_targets.append(labels)
            all_probs.append(probs)
    
    # Объединение всех предсказаний
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    
    # Вычисление метрик (только если есть метки)
    if has_labels:
        metrics = calculate_metrics(all_predictions, all_targets, num_classes, class_names)
    else:
        # Создаем фиктивные метрики для случая без меток
        metrics = {
            'accuracy': 0.0,
            'mean_iou': 0.0,
            'per_class_iou': [0.0] * num_classes,
            'confusion_matrix': None,
            'classification_report': None
        }
    
    return metrics, all_predictions, all_targets, all_probs


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели PointNet')
    parser.add_argument('--test_data', type=str, default='LiDAR/Mar16_test.txt',
                       help='Путь к тестовому набору данных')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту модели')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Количество точек в облаке')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Путь для сохранения результатов')
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Использование устройства: {device}')
    
    # Загрузка чекпоинта
    print(f'Загрузка модели из {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    num_classes = checkpoint['num_classes']
    num_features = checkpoint['num_features']
    
    # Создание модели
    model = PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'Модель загружена. Классов: {num_classes}, Признаков: {num_features}')
    
    # Загрузка тестовых данных
    print('Загрузка тестовых данных...')
    # Проверяем, есть ли метки в тестовом файле
    import pandas as pd
    test_data_sample = pd.read_csv(args.test_data, sep='\t', nrows=1)
    has_labels = 'Classification' in test_data_sample.columns
    
    test_dataset = LiDARDataset(
        args.test_data,
        num_points=args.num_points,
        augment=False,
        has_labels=has_labels
    )
    
    # Если меток нет, используем количество классов из модели
    if not has_labels:
        test_dataset.num_classes = num_classes
        print(f"Тестовый набор без меток. Используется {num_classes} классов из модели.")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Тестирование
    print('Тестирование модели...')
    metrics, predictions, targets, probs = test_model(
        model, test_loader, device, num_classes, has_labels=has_labels
    )
    
    # Вывод результатов
    print('\n' + '='*50)
    print('РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ')
    print('='*50)
    
    if has_labels:
        print(f'Точность (Accuracy): {metrics["accuracy"]:.4f}')
        print(f'Средний IoU (mIoU): {metrics["mean_iou"]:.4f}')
        print('\nIoU по классам:')
        for i, iou in enumerate(metrics['per_class_iou']):
            print(f'  Класс {i}: {iou:.4f}')
        
        print('\nДетальный отчет:')
        print(classification_report(
            targets.cpu().numpy().flatten(),
            predictions.cpu().numpy().flatten(),
            labels=list(range(num_classes)),
            target_names=[f'Class {i}' for i in range(num_classes)],
            zero_division=0
        ))
    else:
        print('Тестовый набор не содержит меток (ground truth).')
        print('Метрики не могут быть вычислены.')
        print(f'Модель обработала {len(predictions.flatten())} точек.')
        print('\nРаспределение предсказанных классов:')
        pred_counts = torch.bincount(predictions.flatten(), minlength=num_classes)
        for i in range(num_classes):
            count = pred_counts[i].item()
            percentage = (count / len(predictions.flatten())) * 100
            print(f'  Класс {i}: {count} точек ({percentage:.2f}%)')
    
    # Визуализация confusion matrix и сохранение результатов
    if args.save_results:
        with open(args.save_results, 'w', encoding='utf-8') as f:
            f.write('РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n')
            f.write('='*50 + '\n')
            
            if has_labels:
                cm_path = args.save_results.replace('.txt', '_confusion_matrix.png')
                plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    [f'Class {i}' for i in range(num_classes)],
                    cm_path
                )
                
                f.write(f'Точность (Accuracy): {metrics["accuracy"]:.4f}\n')
                f.write(f'Средний IoU (mIoU): {metrics["mean_iou"]:.4f}\n\n')
                f.write('IoU по классам:\n')
                for i, iou in enumerate(metrics['per_class_iou']):
                    f.write(f'  Класс {i}: {iou:.4f}\n')
                f.write('\nДетальный отчет:\n')
                f.write(classification_report(
                    targets.cpu().numpy().flatten(),
                    predictions.cpu().numpy().flatten(),
                    labels=list(range(num_classes)),
                    target_names=[f'Class {i}' for i in range(num_classes)],
                    zero_division=0
                ))
            else:
                f.write('Тестовый набор не содержит меток (ground truth).\n')
                f.write('Метрики не могут быть вычислены.\n')
                f.write(f'Модель обработала {len(predictions.flatten())} точек.\n\n')
                f.write('Распределение предсказанных классов:\n')
                pred_counts = torch.bincount(predictions.flatten(), minlength=num_classes)
                for i in range(num_classes):
                    count = pred_counts[i].item()
                    percentage = (count / len(predictions.flatten())) * 100
                    f.write(f'  Класс {i}: {count} точек ({percentage:.2f}%)\n')
    
    print('\nТестирование завершено!')


if __name__ == '__main__':
    main()

