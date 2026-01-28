import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import LiDARDataset
from src.models import (
    PointNetSegmentation,
    PointNetPlusPlusSegmentation,
    DGCNNSegmentation,
    LDGCNNSegmentation,
)


def build_model(model_type, num_classes, num_features):
    if model_type == "pointnet":
        return PointNetSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "pointnet++":
        return PointNetPlusPlusSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "dgcnn":
        return DGCNNSegmentation(num_classes=num_classes, num_features=num_features)
    if model_type == "ldgcnn":
        return LDGCNNSegmentation(num_classes=num_classes, num_features=num_features)
    raise ValueError(f"Неизвестный тип модели: {model_type}")


def save_predictions(model, test_loader, device, output_file, original_data_file, model_type="pointnet"):
    model.eval()
    all_predictions = []
    batch_start_idx = 0

    print("Генерация предсказаний")
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features = features.float().to(device)

            if model_type == "pointnet":
                predictions, _, _ = model(features)
            else:
                predictions = model(features)

            pred_classes = torch.argmax(predictions, dim=2)  # (B, N)

            batch_size, num_points = pred_classes.shape
            for b in range(batch_size):
                start_idx = batch_start_idx + b * num_points
                end_idx = start_idx + num_points
                all_predictions.extend(pred_classes[b].cpu().numpy())
            batch_start_idx += batch_size * num_points

            if (batch_idx + 1) % 10 == 0:
                print(f"Обработано батчей: {batch_idx + 1}/{len(test_loader)}")

    print(f"Всего предсказаний: {len(all_predictions)}")

    print("Загрузка исходных данных...")
    original_data = pd.read_csv(original_data_file, sep="\t")
    print(f"Исходных точек: {len(original_data)}")

    num_original = len(original_data)
    if len(all_predictions) > num_original:
        print(f"Предупреждение: предсказаний больше чем исходных точек. Берем первые {num_original}.")
        all_predictions = all_predictions[:num_original]

    result_data = original_data.copy()
    result_data["Predicted_Classification"] = all_predictions[:len(result_data)]

    print(f"Сохранение результатов в {output_file}...")
    result_data.to_csv(output_file, sep="\t", index=False)
    print("Готово")

    print("\nСтатистика предсказаний:")
    pred_counts = pd.Series(all_predictions[:len(result_data)]).value_counts().sort_index()
    for cls, count in pred_counts.items():
        percentage = (count / len(result_data)) * 100
        print(f"  Класс {cls}: {count} точек ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="Сохранение предсказаний модели")
    parser.add_argument("--checkpoint", type=str, required=True, help="Путь к чекпоинту модели")
    parser.add_argument("--test_data", type=str, required=True, help="Путь к тестовому набору данных")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения результатов")
    parser.add_argument("--num_points", type=int, default=4096, help="Количество точек в облаке")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Использование устройства: {device}")

    print(f"Загрузка модели из {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = checkpoint["num_classes"]
    num_features = checkpoint["num_features"]
    model_type = checkpoint.get("model_type", "pointnet")
    task = checkpoint.get("task", "segmentation")

    if task != "segmentation":
        raise ValueError("save_predictions поддерживает только task=segmentation")

    model = build_model(model_type, num_classes, num_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dataset = LiDARDataset(args.test_data, num_points=args.num_points, augment=False, has_labels=False, task="segmentation")
    test_dataset.num_classes = num_classes

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    save_predictions(model, test_loader, device, args.output, args.test_data, model_type=model_type)


if __name__ == "__main__":
    main()
