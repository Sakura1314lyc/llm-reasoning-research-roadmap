"""Transformer 课程 11：自回归生成与常用采样策略。

每一步只读取最后一个位置的 logits，选出一个新 token 并追加到上下文：

prompt -> model -> next-token logits -> 筛选/采样 -> append -> repeat

本章实现 greedy、temperature、top-k、top-p、重复惩罚和停止字符。
运行前先执行第 10 章，生成模型 checkpoint。
"""

import argparse
from pathlib import Path

import torch

from mini_gpt import GPTConfig, MiniGPT
from mini_gpt_data import CharacterTokenizer, DEFAULT_PROMPT


MODULE_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = (
    MODULE_ROOT / "outputs" / "mini-gpt" / "mini-gpt-character.pth"
)


def load_model_and_tokenizer(
    checkpoint_path: Path,
    device: torch.device
) -> tuple[MiniGPT, CharacterTokenizer, dict]:
    """从第 10 章保存的 checkpoint 恢复配置、词表和模型参数。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"没有找到 checkpoint：{checkpoint_path.resolve()}\n"
            "请先运行：python 02-transformer/10-train-mini-gpt.py"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    config = GPTConfig(**checkpoint["model_config"])
    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_token_ids: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """降低已出现 token 的分数；1.0 表示不使用惩罚。"""
    if penalty < 1.0:
        raise ValueError("repetition_penalty 必须大于或等于 1.0")
    if penalty == 1.0:
        return logits

    adjusted = logits.clone()

    for batch_index in range(generated_token_ids.size(0)):
        used_token_ids = generated_token_ids[batch_index].unique()
        used_logits = adjusted[batch_index, used_token_ids]

        adjusted[batch_index, used_token_ids] = torch.where(
            used_logits < 0,
            used_logits * penalty,
            used_logits / penalty
        )

    return adjusted


def apply_top_k(
    logits: torch.Tensor,
    top_k: int | None
) -> torch.Tensor:
    """只保留分数最高的 k 个候选 token。"""
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    actual_top_k = min(top_k, logits.size(-1))
    threshold = torch.topk(
        logits,
        k=actual_top_k,
        dim=-1
    ).values[:, -1, None]

    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(
    logits: torch.Tensor,
    top_p: float | None
) -> torch.Tensor:
    """保留累计概率达到 p 所需的最小候选集合（nucleus sampling）。"""
    if top_p is None:
        return logits
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须位于 (0, 1] 区间")

    sorted_logits, sorted_indices = torch.sort(
        logits,
        dim=-1,
        descending=True
    )
    cumulative_probabilities = torch.softmax(
        sorted_logits,
        dim=-1
    ).cumsum(dim=-1)

    remove_mask = cumulative_probabilities > top_p

    # 保留第一个使累计概率超过 top_p 的 token，避免候选集合为空。
    remove_mask[:, 1:] = remove_mask[:, :-1].clone()
    remove_mask[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return filtered_logits


@torch.no_grad()
def generate_token_ids(
    model: MiniGPT,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    strategy: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_token_id: int | None = None
) -> torch.Tensor:
    """使用 greedy 或 sampling 策略逐 token 生成。"""
    if prompt_token_ids.ndim != 2 or prompt_token_ids.size(1) == 0:
        raise ValueError("prompt_token_ids 必须是非空的 [B, T] 张量")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能小于 0")
    if strategy not in {"greedy", "sample"}:
        raise ValueError("strategy 必须是 greedy 或 sample")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")

    generated = prompt_token_ids
    model.eval()

    for _ in range(max_new_tokens):
        context = generated[:, -model.config.max_sequence_length:]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]
        next_token_logits = apply_repetition_penalty(
            next_token_logits,
            generated,
            repetition_penalty
        )

        if strategy == "greedy":
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            next_token_logits = apply_top_k(next_token_logits, top_k)
            next_token_logits = apply_top_p(next_token_logits, top_p)
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

        if (
            stop_token_id is not None
            and torch.all(next_token == stop_token_id)
        ):
            break

    return generated


def generate_text(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_character: str | None = None
) -> str:
    """把文本编码、生成 token，再解码回字符串。"""
    if not prompt:
        raise ValueError("prompt 不能为空")

    unknown_characters = sorted(
        set(prompt) - set(tokenizer.token_to_id)
    )
    if unknown_characters:
        raise ValueError(
            "prompt 包含训练词表之外的字符："
            + " ".join(unknown_characters)
        )

    device = next(model.parameters()).device
    prompt_token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
        device=device
    )

    stop_token_id = None
    if stop_character is not None:
        if len(stop_character) != 1:
            raise ValueError("stop_character 必须是单个字符")
        if stop_character not in tokenizer.token_to_id:
            raise ValueError("stop_character 不在训练词表中")
        stop_token_id = tokenizer.token_to_id[stop_character]

    generated = generate_token_ids(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        stop_token_id=stop_token_id
    )

    return tokenizer.decode(generated[0].cpu().tolist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Mini GPT 自回归生成文本")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument(
        "--strategy",
        choices=("greedy", "sample", "both"),
        default="both"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--stop-character", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前环境不可用 CUDA")
        device = torch.device(args.device)

    model, tokenizer, checkpoint = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        device=device
    )

    print("加载 checkpoint：", args.checkpoint.resolve())
    print("训练步数：", checkpoint["training_steps"])
    print("训练最终 loss：", checkpoint["final_loss"])
    print("输入提示：", args.prompt)

    if args.strategy in {"greedy", "both"}:
        greedy_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy="greedy",
            repetition_penalty=args.repetition_penalty,
            stop_character=args.stop_character
        )
        print("\nGreedy：\n", greedy_text)

    if args.strategy in {"sample", "both"}:
        sampled_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            strategy="sample",
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_character=args.stop_character
        )
        print("\nSampling：\n", sampled_text)


if __name__ == "__main__":
    main()
