"""Совместимость со старым именем скрипта.

Этот файл оставлен как тонкая прокладка для старых команд вида
`python scripts/vizualization.py ...`.
Основной скрипт теперь: `scripts/visualization.py`.
"""

import sys
from pathlib import Path

# Добавляем директорию scripts/ в путь, чтобы импорт работал
# как при запуске из корня проекта, так и напрямую из scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from visualization import main  # noqa: E402


if __name__ == "__main__":
    main()
