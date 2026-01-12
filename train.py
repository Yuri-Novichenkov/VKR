import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from models.pointnet import PointNetSegmentation
from data.dataset import LiDARDataset


def calculate_metrics(predictions, targets, num_classes):
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
    
    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'per_class_iou': ious,
        'confusion_matrix': cm
    }


def train_epoch(model, train_loader, optimizer, device, num_classes, lambda_reg=0.001):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.float().to(device)
        labels = labels.long().to(device)
        
        predictions, transform_coords, transform_features = model(features)
        
        # Вычисление потерь
        loss, ce_loss, reg_loss = model.get_loss(
            predictions, labels, transform_coords, transform_features, lambda_reg
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Сохранение предсказаний для метрик
        pred_classes = torch.argmax(predictions, dim=2)
        all_predictions.append(pred_classes)
        all_targets.append(labels)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce_loss': f'{ce_loss.item():.4f}',
            'reg_loss': f'{reg_loss.item():.4f}'
        })
    
    # Вычисление метрик
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate(model, val_loader, device, num_classes):
    """
    Валидация модели
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for features, labels in pbar:
            features = features.float().to(device)
            labels = labels.long().to(device)

            predictions, transform_coords, transform_features = model(features)

            loss, ce_loss, reg_loss = model.get_loss(
                predictions, labels, transform_coords, transform_features
            )
            
            total_loss += loss.item()

            pred_classes = torch.argmax(predictions, dim=2)
            all_predictions.append(pred_classes)
            all_targets.append(labels)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions, all_targets, num_classes)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Обучение PointNet для семантической сегментации')
    parser.add_argument('--train_data', type=str, default='LiDAR/Mar16_train.txt',
                       help='Путь к обучающему набору данных')
    parser.add_argument('--val_data', type=str, default='LiDAR/Mar16_val.txt',
                       help='Путь к валидационному набору данных')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Количество точек в облаке')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Скорость обучения')
    parser.add_argument('--num_classes', type=int, default=None,
                       help='Количество классов')
    parser.add_argument('--lambda_reg', type=float, default=0.001,
                       help='Коэффициент регуляризации трансформаций')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Директория для сохранения моделей')
    parser.add_argument('--resume', type=str, default=None,
                       help='Путь к чекпоинту для возобновления обучения')
    
    args = parser.parse_args()
    
    # Устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'device: {device}')
    else:
        device = torch.device('cpu')
        print('pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
    
    # Создание директории для сохранения
    os.makedirs(args.save_dir, exist_ok=True)
    
    print('Загрузка данных')
    train_dataset = LiDARDataset(
        args.train_data,
        num_points=args.num_points,
        augment=True
    )
    
    val_dataset = LiDARDataset(
        args.val_data,
        num_points=args.num_points,
        augment=False
    )
    
    if args.num_classes is None:
        num_classes = train_dataset.num_classes
    else:
        num_classes = args.num_classes
    
    print(f'Количество классов: {num_classes}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Пропускаем последний неполный батч для избежания проблем с BatchNorm
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Также пропускаем последний неполный батч для консистентности
    )
    
    # Модель
    num_features = len(train_dataset.use_features)
    model = PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    model = model.to(device)
    
    print(f'Модель создана Параметров: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    start_epoch = 0
    best_val_iou = 0
    
    if args.resume:
        print(f'Загрузка чекпоинта из {args.resume}')
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_iou = checkpoint.get('best_val_iou', 0)
        print(f'Возобновление с эпохи {start_epoch}')
    
    # Обучение
    print('Начало обучения')
    for epoch in range(start_epoch, args.epochs):
        print(f'\nЭпоха {epoch + 1}/{args.epochs}')
        print('-' * 50)
        
        # Обучение
        train_metrics = train_epoch(model, train_loader, optimizer, device, 
                                   num_classes, args.lambda_reg)
        
        # Валидация
        val_metrics = validate(model, val_loader, device, num_classes)
        
        # Обновление learning rate
        scheduler.step()
        
        # Вывод метрик
        print(f'\nTrain - Loss: {train_metrics["loss"]:.4f}, '
              f'Accuracy: {train_metrics["accuracy"]:.4f}, '
              f'mIoU: {train_metrics["mean_iou"]:.4f}')
        print(f'Val   - Loss: {val_metrics["loss"]:.4f}, '
              f'Accuracy: {val_metrics["accuracy"]:.4f}, '
              f'mIoU: {val_metrics["mean_iou"]:.4f}')
        
        # Сохранение лучшей модели
        if val_metrics['mean_iou'] > best_val_iou:
            best_val_iou = val_metrics['mean_iou']
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'num_classes': num_classes,
                'num_features': num_features,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Сохранена лучшая модель (mIoU: {best_val_iou:.4f})')
        
        # Сохранение последнего чекпоинта
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_iou': best_val_iou,
            'num_classes': num_classes,
            'num_features': num_features
        }
        torch.save(checkpoint, os.path.join(args.save_dir, 'last_checkpoint.pth'))


if __name__ == '__main__':
    main()

