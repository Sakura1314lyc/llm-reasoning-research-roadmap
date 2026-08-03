"""课程 02：PyTorch 广播机制。

通过多个形状示例理解维度从右向左匹配的广播规则。
"""

# 支持广播的运算可以自动把 Tensor 扩展到兼容的大小，通常不需要真正复制数据。

#满足下列三个条件之一即可广播
# 首先从右往左对应每一对维度
# 1 二者相同 2 二者有一个为1  3 二者有一个没有这个维度

import torch
#例1
x = torch.tensor([
    [1, 2, 3],
    [2, 3 ,4]
])

y = torch.tensor([10, 20, 30])
print("x + y = ", x + y)

#例2
x1 = torch.tensor([
    [1],
    [2],
    [3]
])
y1 = torch.tensor([
    [1, 2, 3, 4]
])

print("x1 + y1 = ", x1 + y1) #有点类似于矩阵乘法变成了元素相加

#例3
print("x + 10 = ", x + 10)

#例4不能广播的情况
t1 = torch.zeros(2, 3)
t2 = torch.ones(2)
#print("t1 + t2 = ", t1 + t2) 会报错，没有满足广播机制

#显式广播
x2 = torch.tensor([10, 20, 30])
y2 = x2.expand(3, 3)
print("y2 = ", y2)
