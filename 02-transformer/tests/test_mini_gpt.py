"""Mini GPT 关键组件的轻量自动化测试。"""

import torch

from mini_gpt import (
    GPTConfig,
    MiniGPT,
    apply_rope,
    build_rope_cache
)
from mini_gpt_data import (
    CharacterTokenizer,
    sample_language_model_batch
)


def build_test_model() -> MiniGPT:
    config = GPTConfig(
        vocab_size=32,
        max_sequence_length=16,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_ff=64,
        dropout=0.0
    )
    return MiniGPT(config)


def test_rope_preserves_shape_and_norm() -> None:
    tensor = torch.randn(2, 4, 6, 8)
    cosine, sine = build_rope_cache(
        sequence_length=6,
        head_dim=8,
        device=tensor.device
    )

    rotated = apply_rope(tensor, cosine, sine)

    assert rotated.shape == tensor.shape
    torch.testing.assert_close(
        rotated.norm(dim=-1),
        tensor.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5
    )


def test_tokenizer_and_shifted_batch() -> None:
    tokenizer = CharacterTokenizer.from_text("春眠不觉晓")
    token_ids = tokenizer.encode("春眠不觉晓")

    assert tokenizer.decode(token_ids) == "春眠不觉晓"

    encoded_text = torch.tensor(token_ids * 10, dtype=torch.long)
    inputs, targets = sample_language_model_batch(
        encoded_text=encoded_text,
        batch_size=4,
        sequence_length=4,
        generator=torch.Generator().manual_seed(42),
        device=torch.device("cpu")
    )

    assert inputs.shape == (4, 4)
    assert targets.shape == (4, 4)
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])


def test_forward_loss_and_backward() -> None:
    model = build_test_model()
    inputs = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))

    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, model.config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_future_tokens_do_not_change_past_logits() -> None:
    model = build_test_model().eval()
    sequence_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    sequence_b = torch.tensor([[1, 2, 3, 9, 8, 7]])

    with torch.no_grad():
        logits_a, _ = model(sequence_a)
        logits_b, _ = model(sequence_b)

    torch.testing.assert_close(
        logits_a[:, :3],
        logits_b[:, :3],
        atol=1e-6,
        rtol=1e-6
    )


def test_generation_appends_requested_tokens() -> None:
    model = build_test_model()
    prompt = torch.tensor([[1, 2, 3]])

    generated = model.generate(
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5
    )

    assert generated.shape == (1, 7)
    torch.testing.assert_close(generated[:, :3], prompt)
