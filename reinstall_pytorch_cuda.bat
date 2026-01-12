@echo off
echo Удаление текущей версии PyTorch...
pip uninstall torch torchvision -y

echo.
echo Установка PyTorch с CUDA 12.1 (стабильная версия)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Проверка установки...
python check_gpu.py

pause

