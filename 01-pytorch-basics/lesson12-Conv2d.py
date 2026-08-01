import torch
from torch import nn

images = torch.randn(
    4, #batch_size
    1, #channels
    28, #高度
    28  #宽度
)

conv = nn.Conv2d(
    in_channels=1, #输入图片只有一个通道
    out_channels=8, #使用八个输出卷积核
    kernel_size=3 ##卷积核大小
)
output = conv(images)
print(images.shape)
print(output.shape)
print(conv.weight.shape)
assert conv.bias is not None
print(conv.bias.shape)
flatten = nn.Flatten() # 转化为二维满足全连接层的形状

x = torch.randn(4, 8, 28, 28)

pool = nn.MaxPool2d(
    kernel_size = 2,
    stride= 2
)

y = pool(x)
print("池化前 : ", x.shape)
print("池化后 : ", y.shape)
z = flatten(x) #默认从第一维开始压平
print("flatten前 : ",x.shape)
print("flatten后 : ",z.shape)

#一个完整的CNN
class simpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#测试形状
model = simpleCNN()

images = torch.randn(
    32,
    1,
    28,
    28
)
logits = model(images)
print("输入形状 : ", images.shape)
print("输出形状 : ", logits.shape)
