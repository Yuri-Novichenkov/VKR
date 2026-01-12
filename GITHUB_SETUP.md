# Инструкция по загрузке проекта на GitHub

## Шаг 1: Подготовка проекта

Проект уже подготовлен с `.gitignore` файлом, который исключает:
- Виртуальное окружение (venv/)
- Большие файлы данных (LiDAR/*.txt, *.laz)
- Модели (checkpoints/*.pth)
- Временные файлы

## Шаг 2: Инициализация Git репозитория

```bash
# Инициализация репозитория
git init

# Добавление всех файлов
git add .

# Первый коммит
git commit -m "Initial commit: PointNet implementation for 3D point cloud segmentation"
```

## Шаг 3: Создание репозитория на GitHub

1. Зайдите на [GitHub.com](https://github.com)
2. Нажмите "New repository" (или перейдите по ссылке: https://github.com/new)
3. Заполните:
   - **Repository name:** `pointnet-lidar-segmentation` (или другое название)
   - **Description:** "PointNet implementation for semantic segmentation of 3D LiDAR point clouds"
   - **Visibility:** Public или Private (на ваше усмотрение)
   - **НЕ** создавайте README, .gitignore или license (они уже есть)
4. Нажмите "Create repository"

## Шаг 4: Подключение к GitHub

```bash
# Добавьте remote репозиторий (замените YOUR_USERNAME на ваш GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/pointnet-lidar-segmentation.git

# Переименуйте ветку в main (если нужно)
git branch -M main

# Загрузите код на GitHub
git push -u origin main
```

## Шаг 5: Загрузка больших файлов (опционально)

Если нужно загрузить данные LiDAR, используйте **Git LFS**:

```bash
# Установите Git LFS (если еще не установлен)
# Windows: скачайте с https://git-lfs.github.com/

# Инициализируйте Git LFS в репозитории
git lfs install

# Отслеживайте большие файлы
git lfs track "LiDAR/*.txt"
git lfs track "*.laz"
git lfs track "*.pth"

# Добавьте .gitattributes
git add .gitattributes

# Коммит и push
git commit -m "Add Git LFS tracking for large files"
git push origin main
```

**Примечание:** Git LFS имеет ограничения на бесплатном аккаунте (1 GB storage, 1 GB bandwidth/month). Для больших файлов рассмотрите альтернативы:
- Google Drive / Dropbox
- Zenodo (для научных данных)
- Указание в README, где скачать данные

## Что будет загружено:

✅ **Будет загружено:**
- Весь исходный код (models/, data/, *.py)
- README.md и документация
- requirements.txt
- .gitignore
- Скрипты установки (install_pytorch.bat, install_pytorch.sh)

❌ **НЕ будет загружено** (из-за .gitignore):
- venv/ (виртуальное окружение)
- LiDAR/*.txt (большие файлы данных)
- checkpoints/*.pth (обученные модели)
- visualizations/*.png (можно добавить позже, если нужны)

## Рекомендации для README на GitHub:

Добавьте в README информацию о том, где скачать данные:
```markdown
## Данные

Данные для обучения можно скачать с [ссылка на источник].
Поместите файлы в директорию `LiDAR/`:
- Mar16_train.txt
- Mar16_val.txt
- Mar16_test.txt
```

## Полезные команды Git:

```bash
# Проверить статус
git status

# Посмотреть, что будет загружено
git ls-files

# Добавить конкретный файл
git add filename.py

# Посмотреть историю коммитов
git log

# Обновить репозиторий после изменений
git add .
git commit -m "Описание изменений"
git push origin main
```

## Troubleshooting

**Проблема:** "remote: error: File is too large"
**Решение:** Используйте Git LFS или исключите файл через .gitignore

**Проблема:** "Permission denied"
**Решение:** Проверьте, что вы авторизованы в Git:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

