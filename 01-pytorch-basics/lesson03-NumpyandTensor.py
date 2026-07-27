#注意下numpy只能用于cpu

import numpy as np
import torch

array  = np.array([
    [1, 2,  3],
    [4, 5, 6]
])

tensor = torch.from_numpy(array) ##这种写法会共享内存, 修改了numpy的值,会影响tensor的值
print(array)
print(type(array))
array[0, 0] = 10
print(tensor)
print(type(tensor))
tensor[1,1] = 10
print(array) #也就是反过来也会受到影响


#若是使用torch.tensor
#只是复制数据,即不会影响原来的数据

array1 = np.array([1, 2, 3])
tensor1 = torch.tensor(array1)
array1[0] = 10
print(array1)
print(tensor1)

#神经网络中常见的转换
array = tensor.detach().cpu().numpy()
# detach() 从计算图中分离
# .cpu() 移动到 CPU
# .numpy()指转变成numpy