# Инструкция по установке PyTorch с поддержкой GPU

## Текущая ситуация
- GPU: NVIDIA GeForce RTX 2070
- Драйвер: 591.74
- CUDA версия драйвера: 13.1

## Проблема
PyTorch не видит CUDA и использует CPU вместо GPU.

## Решение

### Вариант 1: Стабильная версия с CUDA 12.1 (рекомендуется)

```powershell
# Удалить текущую версию
pip uninstall torch torchvision -y

# Установить стабильную версию с CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Вариант 2: Стабильная версия с CUDA 11.8 (более совместимая)

```powershell
# Удалить текущую версию
pip uninstall torch torchvision -y

# Установить стабильную версию с CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Вариант 3: Оставить nightly с CUDA 13.0 (если нужно)

Если хотите оставить nightly версию, убедитесь что она правильно установлена:

```powershell
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

## Проверка

После установки выполните:

```powershell
python check_gpu.py
```

Должно вывести:
- CUDA available: True
- GPU name: NVIDIA GeForce RTX 2070

## Запуск обучения

После успешной установки:

```powershell
python train.py --train_data LiDAR/Mar16_train.txt --val_data LiDAR/Mar16_val.txt
```

Теперь должно использоваться GPU вместо CPU.

