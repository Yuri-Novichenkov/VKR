#!/bin/bash

echo "========================================"
echo "Установка PyTorch с CUDA 13.0 (Nightly)"
echo "========================================"
echo ""

echo "Удаление существующей версии PyTorch (если есть)..."
pip uninstall torch torchvision -y

echo ""
echo "Установка PyTorch Nightly с CUDA 13.0..."
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

echo ""
echo "========================================"
echo "Проверка установки..."
echo "========================================"
python -c "import torch; print(f'PyTorch версия: {torch.__version__}'); print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Установка завершена!"

