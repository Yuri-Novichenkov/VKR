"""
Скрипт для проверки доступности CUDA
"""

import torch

print("=" * 50)
print("ПРОВЕРКА CUDA")
print("=" * 50)

print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"cuDNN версия: {torch.backends.cudnn.version()}")
    print(f"Количество GPU: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Название: {torch.cuda.get_device_name(i)}")
        print(f"  Память: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Тест на GPU
    print("\nТест вычислений на GPU:")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("  ✓ Успешно! GPU работает корректно.")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
else:
    print("\nCUDA недоступна. Возможные причины:")
    print("1. Установлена CPU-версия PyTorch")
    print("2. Драйверы NVIDIA не установлены или устарели")
    print("3. CUDA toolkit не установлен")
    print("\nДля установки PyTorch с CUDA:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")

print("=" * 50)

