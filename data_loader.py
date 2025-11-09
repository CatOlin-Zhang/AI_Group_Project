# 检查GPU是否可用
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")
#pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126