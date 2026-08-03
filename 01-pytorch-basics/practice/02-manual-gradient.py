import torch

x = torch.tensor(2.0)

w = torch.tensor(2.0, requires_grad=True) # True的含义表示跟踪这个w, 使得后续梯度下降时可以计算这个w的梯度

b = torch.tensor(3.0, requires_grad=True) 
y = torch.tensor(8.0)
print("w = ", w.item())
# 前向传播
y_pre = x * w + b

# 计算损失
error = y_pre - y
Loss = (y_pre - y) ** 2

print("预测值为", y_pre)
print("损失为", Loss)

# 反向传播

lr = 0.01
Loss.backward()
print("w 梯度", w.grad)
print("b 梯度", b.grad)
print("更新后的w", w - lr * error * 2 * x)
print("更新后的b", b - lr * error * 2)

#叶子与非叶子
print(w.is_leaf)
print(b.is_leaf)
print(y_pre.is_leaf)
print(Loss.is_leaf)