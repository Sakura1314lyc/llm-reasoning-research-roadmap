"""用固定 logits 比较 Greedy、Temperature、Top-k 和 Top-p。"""

import torch


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float = 1.0) -> torch.Tensor:
    """过滤一维 logits；被移除的候选设为负无穷。"""
    if logits.ndim != 1:
        raise ValueError("logits 必须是一维张量")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须位于 (0, 1]")
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits = logits.clone()
        logits[sorted_indices[remove]] = float("-inf")
    return logits


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float = 1.0,
    seed: int = 42,
) -> int:
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    generator = torch.Generator().manual_seed(seed)
    filtered = filter_logits(logits / temperature, top_k=top_k, top_p=top_p)
    return int(torch.multinomial(torch.softmax(filtered, dim=-1), 1, generator=generator).item())


if __name__ == "__main__":
    scores = torch.tensor([4.0, 2.0, 1.0, 0.5])
    print("greedy token:", int(scores.argmax().item()))
    for name, kwargs in {
        "temperature": {"temperature": 0.7},
        "top-k": {"top_k": 2},
        "top-p": {"top_p": 0.8},
    }.items():
        print(f"{name} token:", sample_token(scores, **kwargs))
