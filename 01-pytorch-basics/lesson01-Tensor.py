# Tensor 翻译为张量 理解为能够在CPU或者GPU上进行高效数学运算的多维数组(本身是一种数据结构)

import torch
print(torch.__version__)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

# 创建Tensor
a = torch.tensor(10)
b = torch.tensor(
    [1, 2, 3],
    dtype=torch.float32
)
c = torch.tensor([
    [1, 2, 3],
    [3, 4, 5]
])
d = torch.tensor([[
    [1, 2,  3],
    [1, 2, 3],
    [1, 2, 3]
]])
print(a)
print(f"a的维度是{a.ndim}")
print(b)
print(f"b的维度是{b.ndim}")
print(c)
print(f"c的维度是{c.ndim}")
print(f"d的维度是{d.ndim}")

#创建全零矩阵
zero = torch.zeros((2, 3)) # 意思形状是二行三列的全零矩阵
print(zero)

#创建全一矩阵
one = torch.ones(2, 3) # 意思是形状是二行三列的全一矩阵
print(one)

#创建随机矩阵(最常用)
random_tensor = torch.rand(2, 3) # 生成的数位于[0, 1]之间
print(random_tensor)

#randn生成的数据符合正态分布
#平均差为一，均值为零(生成数可能有负)
normal_tensor = torch.randn(2, 3)
print(normal_tensor)

#创建整数序列(用法类似于python的range)
x = torch.arange(0, 10)
y = torch.arange(0, 10, 2)
print(f"x = f{x}")
print(f"y = f{y}")

#创建等间距数据
t1 = torch.linspace(0, 1, 5) #意思是从0到1等长生成5个数
print(f"t1 = {t1}")
t2 = torch.linspace(1, 5, 10)
print(f"t2 = {t2}")

#torch关注三要素: shape dtype device
temp1 = torch.tensor([
    [1, 2, 3],
    [2, 3, 4]
],  dtype=torch.float32
)

X =  torch.arange(12)
print("X = " ,X)
X = X.reshape(3, 4)
print("reshape 后 X = ", X)
print(f"temp1 = {temp1}")
print("temp1的形状为", temp1.shape)
print("temp1的数据类型为", temp1.dtype)
temp1 = temp1.to(device)
print("temp1的设备为", temp1.device)


a1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float32
)

# 对于学习过c++的学者来说,这里的tensor特别好理解，就把他看作数组去做就行
t3 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]   
])

print(t3[0]) #和数组一模一样的访问下标方法
# 介绍下取整行
print(t3[0, :]) # :表示取这一维的全部元素
# 取整列
print(t3[:,  0])

# 取区域
print(t3[0 : 2, 1 : 3]) # 左开右闭


#基本运算
x1 = torch.tensor([
    [1, 2],
    [3, 4]
])

x2 = torch.tensor([
    [5, 6],
    [7, 8]
])

x3 = torch.tensor([1, 2])
x4 = torch.tensor([3, 4])
print("x1 + x2 = ", x1 + x2)
print("x1 - x2 = ", x1 - x2)
print("x1 * x2 = ", x1 * x2)
print("x1 / x2 = ", x1 / x2)
# 点积(也为内积)
result = torch.dot(x3, x4)
print("x3 点乘 x4 = ", result)
# 矩阵乘法(n * m @ m * k = n * k)
x5 = torch.tensor([
    [1, 2 ,3],
    [4, 5, 6]
], dtype= torch.float32
)
x6 = torch.tensor([
    [1, 2],
    [3, 4],
    [5, 6]
], dtype=torch.float32)
print("x5 @ x6 = ", x5 @ x6)
print("x5 @ x6 = ", torch.matmul(x5, x6))


# 聚合运算(sum, mean, max, min)
print("x5 sum = ", x5.sum())
print("x5 mean = ", x5.mean())
print("x5 max = ", x5.max())
print("x5 min = ", x5.min())
#也可以指定维度
print("x5 sum1 = ", x5.sum(dim = 1)) #dim=n 表示把第 n 个维度压缩掉。

#增加维度 unsqueeze
temp2 = torch.tensor([1, 2,3 ])
y = temp2.unsqueeze(0) #增加第0维 
print("y = ", y)
y = temp2.unsqueeze(1) #增加第1维
print("y = ", y)
#降维度
temp3 = torch.zeros(1, 2,3 ,4)
print("temp3 shape is ", temp3.shape)
temp4 = temp3.squeeze() #当没指定参数时, 默认删除1维度
print("temp4 shape is ", temp4.shape) 
temp5 = temp3.squeeze(dim = 0)
print("temp5 shape is ", temp5.shape)