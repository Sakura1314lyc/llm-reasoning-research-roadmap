"""Transformer 课程 10：在字符级小文本上训练 Mini GPT。

本章重点不是模型效果，而是走通语言模型训练闭环：

文本 -> 字符 token -> 随机连续片段 -> 右移标签 -> Causal LM Loss
     -> 反向传播 -> 梯度裁剪 -> AdamW 更新 -> 保存 checkpoint

默认语料是一首重复的小诗，模型应能快速过拟合，证明数据流和训练逻辑正确。
"""

import argparse
from dataclasses import asdict
from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch

from mini_gpt import GPTConfig, MiniGPT
from mini_gpt_data import (
    CharacterTokenizer,
    TRAINING_CORPUS,
    sample_language_model_batch
)


SEED = 42
BATCH_SIZE = 16
SEQUENCE_LENGTH = 16
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0

MODULE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_ROOT / "outputs" / "mini-gpt"
CHECKPOINT_PATH = OUTPUT_DIR / "mini-gpt-character.pth"
LOSS_CURVE_PATH = OUTPUT_DIR / "training-loss.png"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用 CUDA")
    return torch.device(device_name)


def build_model(tokenizer: CharacterTokenizer) -> MiniGPT:
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=SEQUENCE_LENGTH,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=96,
        dropout=0.0
    )
    return MiniGPT(config)


def train(
    model: MiniGPT,
    encoded_text: torch.Tensor,
    steps: int,
    device: torch.device
) -> list[float]:
    """训练指定步数并返回每一步的 loss。"""
    if steps <= 0:
        raise ValueError("steps 必须大于 0")

    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    batch_generator = torch.Generator().manual_seed(SEED)
    loss_history: list[float] = []
    log_interval = max(1, steps // 10)

    for step in range(1, steps + 1):
        inputs, targets = sample_language_model_batch(
            encoded_text=encoded_text,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            generator=batch_generator,
            device=device
        )

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)

        if loss is None:
            raise RuntimeError("传入 targets 后模型没有返回 loss")

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_NORM
        )

        optimizer.step()
        loss_history.append(loss.item())

        if step == 1 or step % log_interval == 0 or step == steps:
            print(
                f"step={step:4d}/{steps} | "
                f"loss={loss.item():.4f} | "
                f"grad_norm={float(gradient_norm):.4f}"
            )

    return loss_history


def save_checkpoint(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    loss_history: list[float],
    checkpoint_path: Path
) -> None:
    """保存恢复推理所需的模型配置、参数和词表。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "training_steps": len(loss_history),
            "final_loss": loss_history[-1]
        },
        checkpoint_path
    )


def plot_loss_curve(
    loss_history: list[float],
    save_path: Path
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        range(1, len(loss_history) + 1),
        loss_history,
        color="tab:blue",
        linewidth=1.5
    )
    axis.set_title("Mini GPT Training Loss")
    axis.set_xlabel("Training Step")
    axis.set_ylabel("Cross-Entropy Loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练字符级 Mini GPT")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存 checkpoint"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不保存 loss 曲线"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    device = resolve_device(args.device)
    tokenizer = CharacterTokenizer.from_text(TRAINING_CORPUS)
    encoded_text = torch.tensor(
        tokenizer.encode(TRAINING_CORPUS),
        dtype=torch.long
    )
    model = build_model(tokenizer)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("训练设备：", device)
    print("语料字符数：", encoded_text.numel())
    print("词表大小：", tokenizer.vocab_size)
    print("模型参数量：", parameter_count)

    loss_history = train(
        model=model,
        encoded_text=encoded_text,
        steps=args.steps,
        device=device
    )

    print("初始 loss：", loss_history[0])
    print("最终 loss：", loss_history[-1])

    if len(loss_history) >= 10:
        first_average = sum(loss_history[:5]) / 5
        last_average = sum(loss_history[-5:]) / 5
        print("前 5 步平均 loss：", first_average)
        print("后 5 步平均 loss：", last_average)
        assert last_average < first_average, "loss 没有下降，请检查训练流程"

    if not args.no_save:
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            loss_history=loss_history,
            checkpoint_path=CHECKPOINT_PATH
        )
        print("checkpoint：", CHECKPOINT_PATH.resolve())

    if not args.no_plot:
        plot_loss_curve(loss_history, LOSS_CURVE_PATH)
        print("loss 曲线：", LOSS_CURVE_PATH.resolve())


if __name__ == "__main__":
    main()
