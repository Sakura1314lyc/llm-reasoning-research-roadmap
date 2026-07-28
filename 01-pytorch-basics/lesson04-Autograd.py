import torch

x = torch.tensor(2.0, requires_grad=True) #若为false, 不会自动为其记录用于求导的计算图
y = 3 * x * x + 4 * x + 5
y.backward()
print(x.grad) #保存的就是dy/dx
print(x.grad_fn)
print(y.grad_fn)
#梯度本身会累加
y1 = x ** 2
y1.backward()
print(x.grad)

#清空梯度
assert x.grad is not None 
x.grad.zero_() #  _表示原地操作

# detach() 会创建一个与原Tensor共享数据, 但从当前计算图中分离出来的tensor
z = y.detach()
print("y.grad = ", y.grad)
print("z.grad = ", z.grad)
print("y.grad.fn = ", y.grad_fn)
print("z.grad.fn = ", z.grad_fn)
print("y is ", y.requires_grad)
print("z is ", z.requires_grad)

# detach控制了一个tensor 不会被记录
# no_grad() 控制的是一段代码

#向量不能调用无参数的backward() ,因为不知道你是想求哪个标量的目标梯度

