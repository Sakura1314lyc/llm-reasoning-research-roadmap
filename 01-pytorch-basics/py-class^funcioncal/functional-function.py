
# 高阶函数
def cal(x : int, y : int):
    return x - y
def ye(x, y, cal):
    return cal(x, y) + cal(y, x)
x = 1
y = 2
print(ye(x, y, cal))

#返回函数
def F(num):
    def cal():
        ax = 0
        for n in num:
            ax += n
        return ax
    return cal

f = F([1, 2, 3, 4, 5, 6])
print(f())

def a():
    x = 0
    def f():
        nonlocal x
        x = x + 1
        return x
    return f
ans1 = a()
ans2 = a()
print(f"{ans1()}和{ans2()}")

#匿名函数
f = lambda x : x * x * x
g = lambda x, y : x * y
print(f"{f(4)}, {g(2, 3)}")

def bulid():
    return lambda x, y : abs(x - y)
f = bulid()
print(f"{f(2, 3)}")

def is_even(x):
    return x % 2 == 0

G1 = list(filter(lambda x : x % 2 == 1, range(1, 20)))
G2 = list(filter(is_even, range(1, 21)))
print(f"{G1}    {G2}")

def show(name : str):
    print(f"hello, {name}")
print(f"{show("yucheng")}, {show.__name__}")


#偏函数
x = "100"
s2i = int(x)
s2ib = int(x, base = 2)
s2id = int(x, base = 8)
print(f"x = {x}, 二进制为{s2ib}, 八进制为{s2id}")