import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    # Тест
    x = torch.randn(10, 10).cuda()
    print("GPU test: OK")
else:
    print("CUDA NOT AVAILABLE - нужно переустановить PyTorch с CUDA")

