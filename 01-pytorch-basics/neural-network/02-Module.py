from typing import Any

import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # nn.Linear(in, out)
        self.linear = nn.Linear(
            in_features=1, # 看的是输入特征
            out_features=1 # 看的是输出特征
        ) # 这里的in, out 是权重转置的形状[in, out], b 的形状是[out]
    def forward(self, x):
        return self.linear(x)

model = LinearModel()
print(model)

for name, parameter in model.named_parameters(): #主要用于查看和遍历可训练参数
    print(name)
    print(parameter)
    print("shape = ", parameter.shape)
    print("required_parameter is ", parameter.requires_grad)
    print()

#查看模型状态
print(model.state_dict()) #主要用于保存和恢复模型状态、

#with torch.no_grad(): 他的意思是不构建自动求导计算图以更新梯度
# model.eval 把部分网路层切换至推理状态,注意推理状态是不会关闭梯度计算的