import torch
from torch import nn

bn = nn.BatchNorm2d(32)
EPOCHS = 500
for name, parameter in  bn.named_parameters():
    print(
        name, 
        parameter.shape
    )
# 构建一个带BN的更强CNN
class BetterFashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # =====================
            # Block 1
            # =====================
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =====================
            # Block 2
            # =====================
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            # =====================
            # Block 3
            # =====================
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), #对每个通道的整个空间区域求平均

            nn.Flatten(),

            nn.Dropout(p=0.2),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

#Learning Rate Scheduler(学习率调度器 ： 先找到最佳学习的一个区域范围然后再用小的学习率进行一个慢慢调)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = BetterFashionCNN().to(device)

loss_fn = nn.CrossEntropyLoss(
    label_smoothing=0.05 # 稍微给别的类别一点概率
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.002,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-5
)

