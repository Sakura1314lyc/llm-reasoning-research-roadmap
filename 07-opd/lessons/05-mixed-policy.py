"""按比例混合 SFT/Off-policy 与 On-policy 蒸馏 Loss。"""

import torch

from distillation_utils import mixed_loss


def main() -> None:
    sft = torch.tensor(1.2)
    online = torch.tensor(0.8)
    for ratio in (0.0, 0.5, 1.0):
        print(ratio, mixed_loss(sft, online, ratio).item())
    assert mixed_loss(sft, online, 0).item() == sft.item()


if __name__ == "__main__":
    main()
