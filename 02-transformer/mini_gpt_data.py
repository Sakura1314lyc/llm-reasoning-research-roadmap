"""Mini GPT 训练和生成共用的字符级数据工具。"""

import torch


BASE_TEXT = "春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。\n"
TRAINING_CORPUS = BASE_TEXT * 80
DEFAULT_PROMPT = "春眠"


class CharacterTokenizer:
    """最小字符级 tokenizer：一个 Unicode 字符对应一个 token。"""

    UNKNOWN_TOKEN = "<unk>"

    def __init__(self, tokens: list[str]) -> None:
        if not tokens or tokens[0] != self.UNKNOWN_TOKEN:
            raise ValueError("tokens 的第一个元素必须是 <unk>")
        if len(tokens) != len(set(tokens)):
            raise ValueError("tokens 中不能存在重复项")

        self.tokens = tokens
        self.token_to_id = {
            token: index
            for index, token in enumerate(tokens)
        }
        self.unknown_token_id = self.token_to_id[self.UNKNOWN_TOKEN]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("训练文本不能为空")

        vocabulary = sorted(set(text))
        return cls([cls.UNKNOWN_TOKEN, *vocabulary])

    @classmethod
    def from_state_dict(cls, state: dict) -> "CharacterTokenizer":
        tokens = state.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("tokenizer 状态中缺少 tokens 列表")
        return cls(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [
            self.token_to_id.get(character, self.unknown_token_id)
            for character in text
        ]

    def decode(self, token_ids: list[int]) -> str:
        characters: list[str] = []

        for token_id in token_ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID 越界：{token_id}")

            token = self.tokens[token_id]
            characters.append("�" if token == self.UNKNOWN_TOKEN else token)

        return "".join(characters)

    def state_dict(self) -> dict[str, list[str]]:
        return {"tokens": self.tokens.copy()}


def sample_language_model_batch(
    encoded_text: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机采样连续片段，并构造右移一位的输入与目标。"""
    if encoded_text.ndim != 1:
        raise ValueError("encoded_text 必须是一维张量")
    if encoded_text.numel() <= sequence_length:
        raise ValueError("文本长度必须大于 sequence_length")

    maximum_start = encoded_text.numel() - sequence_length - 1
    start_indices = torch.randint(
        low=0,
        high=maximum_start + 1,
        size=(batch_size,),
        generator=generator
    )

    windows = torch.stack([
        encoded_text[start:start + sequence_length + 1]
        for start in start_indices.tolist()
    ])

    inputs = windows[:, :-1].to(device)
    targets = windows[:, 1:].to(device)

    return inputs, targets
