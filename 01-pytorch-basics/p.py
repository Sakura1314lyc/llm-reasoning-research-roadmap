import torch

x = torch.arange(1, 7).reshape(2, 3).float()

weight = torch.tensor([
    [1., 2.],
    [3., 4.],
    [5., 6.]
])

bias = torch.tensor([10., 20.])


# 输出 x、weight、bias 的形状；
# 计算 x @ weight；
# 判断结果形状；
# 再加上 bias；
# 手动计算最终结果，检查是否与 PyTorch 一致；
# 输出 weight.T 及其形状。
print(x.shape)
print(weight.shape)
print(bias.shape)
print(x @ weight)
ans= x @ weight + bias
print(ans)
print(weight.T, weight.shape)