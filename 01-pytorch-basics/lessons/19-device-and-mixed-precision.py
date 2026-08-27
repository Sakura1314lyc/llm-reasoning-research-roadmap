"""把模型和数据放到同一设备，并演示自动混合精度。"""

import torch
from torch import nn


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    inputs = torch.randn(4, 8, device=device)
    targets = torch.randint(0, 2, (4,), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    # 只有 CUDA 才启用 autocast/GradScaler；CPU 路径仍可直接验证。
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print("设备：", device)
    print("logits / loss：", logits.shape, loss.item())
    if device.type == "cuda":
        print("GPU：", torch.cuda.get_device_name(device))
        print("峰值显存 MB：", torch.cuda.max_memory_allocated(device) / 1024**2)

    assert logits.device == device
    assert torch.isfinite(loss)


if __name__ == "__main__":
    main()
