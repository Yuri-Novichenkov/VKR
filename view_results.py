import torch
import argparse
import os

def view_checkpoint(checkpoint_path):
    """
    Просмотр содержимого чекпоинта
    """
    if not os.path.exists(checkpoint_path):
        print(f"aайл {checkpoint_path} не найден")
        return
    
    print(f"\n{'='*60}")
    print(f"Анализ чекпоинта: {checkpoint_path}")
    print(f"{'='*60}\n")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("Основная информация:")
    print(f"  Эпоха: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Количество классов: {checkpoint.get('num_classes', 'N/A')}")
    print(f"  Количество признаков: {checkpoint.get('num_features', 'N/A')}")
    print(f"  Лучший валидационный mIoU: {checkpoint.get('best_val_iou', 'N/A'):.4f}")
    
    if 'train_metrics' in checkpoint:
        print("\nМетрики обучения (лучшая модель):")
        train_metrics = checkpoint['train_metrics']
        print(f"  Loss: {train_metrics.get('loss', 'N/A'):.4f}")
        print(f"  Accuracy: {train_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  mIoU: {train_metrics.get('mean_iou', 'N/A'):.4f}")
        
        if 'per_class_iou' in train_metrics:
            print("\n  IoU по классам (обучение):")
            for i, iou in enumerate(train_metrics['per_class_iou']):
                print(f"    Класс {i}: {iou:.4f}")
    
    if 'val_metrics' in checkpoint:
        print("\nМетрики валидации (лучшая модель):")
        val_metrics = checkpoint['val_metrics']
        print(f"  Loss: {val_metrics.get('loss', 'N/A'):.4f}")
        print(f"  Accuracy: {val_metrics.get('accuracy', 'N/A'):.4f}")
        print(f"  mIoU: {val_metrics.get('mean_iou', 'N/A'):.4f}")
        
        if 'per_class_iou' in val_metrics:
            print("\n  IoU по классам (валидация):")
            for i, iou in enumerate(val_metrics['per_class_iou']):
                print(f"    Класс {i}: {iou:.4f}")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Просмотр результатов обучения')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Путь к чекпоинту (по умолчанию: checkpoints/best_model.pth)')
    parser.add_argument('--all', action='store_true',
                       help='Показать все чекпоинты')
    
    args = parser.parse_args()
    
    if args.all:
        checkpoint_dir = os.path.dirname(args.checkpoint) if os.path.dirname(args.checkpoint) else 'checkpoints'
        best_model = os.path.join(checkpoint_dir, 'best_model.pth')
        last_checkpoint = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
        
        if os.path.exists(best_model):
            view_checkpoint(best_model)
        
        if os.path.exists(last_checkpoint):
            view_checkpoint(last_checkpoint)
    else:
        view_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()

