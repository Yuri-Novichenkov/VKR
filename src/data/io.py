"""
Утилиты ввода/вывода для облаков точек.

Поддерживаются форматы:
- .txt / .tsv  — tab-separated таблица с заголовком;
- .las / .laz  — бинарный LiDAR-формат (требует laspy + lazrs).
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataframe(data_path) -> pd.DataFrame:
    """Загружает облако точек в DataFrame независимо от формата файла.

    Колонки для LAS/LAZ: X, Y, Z и по наличию R, G, B, Intensity,
    NumberOfReturns, ReturnNumber, Classification.

    Args:
        data_path: str или Path — путь к файлу.

    Returns:
        pd.DataFrame с данными облака точек.

    Raises:
        ImportError: если .las/.laz, но laspy не установлен.
    """
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix in (".laz", ".las"):
        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "Для чтения .laz/.las установите laspy и lazrs: "
                "pip install laspy lazrs"
            ) from exc

        las = laspy.read(str(path))
        data = {
            "X": np.asarray(las.x),
            "Y": np.asarray(las.y),
            "Z": np.asarray(las.z),
        }
        if hasattr(las, "red"):
            data["R"] = np.asarray(las.red)
        if hasattr(las, "green"):
            data["G"] = np.asarray(las.green)
        if hasattr(las, "blue"):
            data["B"] = np.asarray(las.blue)
        if hasattr(las, "intensity"):
            data["Intensity"] = np.asarray(las.intensity)
        if hasattr(las, "number_of_returns"):
            data["NumberOfReturns"] = np.asarray(las.number_of_returns)
        if hasattr(las, "return_number"):
            data["ReturnNumber"] = np.asarray(las.return_number)
        if hasattr(las, "classification"):
            data["Classification"] = np.asarray(las.classification)
        return pd.DataFrame(data)

    # Текстовые форматы: tab-separated (стандарт для .txt/.tsv из LiDAR-ПО)
    return pd.read_csv(path, sep="\t")
