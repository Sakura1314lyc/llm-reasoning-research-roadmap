import numpy as np
import torch
import transformers

print("NumPy 版本：", np.__version__)
print("PyTorch 版本：", torch.__version__)
print("Transformers 版本：", transformers.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前计算设备：", device)

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print("测试张量：", x)