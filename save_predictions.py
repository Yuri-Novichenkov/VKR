"""
Скрипт для сохранения предсказаний модели в файл
"""

import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader

from models.pointnet import PointNetSegmentation
from data.dataset import LiDARDataset


def save_predictions(model, test_loader, device, output_file, original_data_file):
    """
    Сохранение предсказаний модели в файл
    """
    model.eval()
    all_predictions = []
    all_indices = []
    batch_start_idx = 0
    
    print('Генерация предсказаний...')
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.float().to(device)
            
            # Forward pass
            predictions, _, _ = model(features)
            
            # Получение предсказанных классов
            pred_classes = torch.argmax(predictions, dim=2)  # (B, N)
            
            # Сохранение предсказаний для этого батча
            batch_size, num_points = pred_classes.shape
            for b in range(batch_size):
                # Получаем индексы точек для этого облака
                # (нужно будет восстановить исходные индексы)
                start_idx = batch_start_idx + b * num_points
                end_idx = start_idx + num_points
                
                all_predictions.extend(pred_classes[b].cpu().numpy())
                all_indices.extend(range(start_idx, end_idx))
            
            batch_start_idx += batch_size * num_points
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Обработано батчей: {batch_idx + 1}/{len(test_loader)}')
    
    print(f'Всего предсказаний: {len(all_predictions)}')
    
    # Загрузка исходных данных
    print('Загрузка исходных данных...')
    original_data = pd.read_csv(original_data_file, sep='\t')
    print(f'Исходных точек: {len(original_data)}')
    
    # Создание DataFrame с предсказаниями
    # Если предсказаний больше чем исходных точек (из-за перекрытия облаков),
    # берем только первые N предсказаний
    num_original = len(original_data)
    if len(all_predictions) > num_original:
        print(f'Предупреждение: предсказаний больше чем исходных точек. Берем первые {num_original}.')
        all_predictions = all_predictions[:num_original]
    
    # Добавляем колонку с предсказаниями
    result_data = original_data.copy()
    result_data['Predicted_Classification'] = all_predictions[:len(result_data)]
    
    # Сохранение результатов
    print(f'Сохранение результатов в {output_file}...')
    result_data.to_csv(output_file, sep='\t', index=False)
    print('Готово!')
    
    # Статистика
    print('\nСтатистика предсказаний:')
    pred_counts = pd.Series(all_predictions[:len(result_data)]).value_counts().sort_index()
    for cls, count in pred_counts.items():
        percentage = (count / len(result_data)) * 100
        print(f'  Класс {cls}: {count} точек ({percentage:.2f}%)')


def main():
    parser = argparse.ArgumentParser(description='Сохранение предсказаний модели')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту модели')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Путь к тестовому набору данных')
    parser.add_argument('--output', type=str, required=True,
                       help='Путь для сохранения результатов')
    parser.add_argument('--num_points', type=int, default=4096,
                       help='Количество точек в облаке')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    
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
    import pandas as pd
    test_data_sample = pd.read_csv(args.test_data, sep='\t', nrows=1)
    has_labels = 'Classification' in test_data_sample.columns
    
    test_dataset = LiDARDataset(
        args.test_data,
        num_points=args.num_points,
        augment=False,
        has_labels=has_labels
    )
    
    if not has_labels:
        test_dataset.num_classes = num_classes
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Сохранение предсказаний
    save_predictions(model, test_loader, device, args.output, args.test_data)


if __name__ == '__main__':
    main()

