"""第 11 章 checkpoint 恢复与采样策略测试。"""

from dataclasses import asdict
import importlib.util
from pathlib import Path

import torch

from mini_gpt import GPTConfig, MiniGPT
from mini_gpt_data import CharacterTokenizer


CHAPTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "11-autoregressive-generation.py"
)


def load_generation_chapter():
    spec = importlib.util.spec_from_file_location(
        "autoregressive_generation_chapter",
        CHAPTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("无法加载第 11 章")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top_k_and_top_p_filter_candidates() -> None:
    chapter = load_generation_chapter()
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])

    top_k_logits = chapter.apply_top_k(logits, top_k=2)
    assert torch.isfinite(top_k_logits).sum().item() == 2

    top_p_logits = chapter.apply_top_p(logits, top_p=0.7)
    assert torch.isfinite(top_p_logits).sum().item() == 2


def test_checkpoint_round_trip_and_greedy_generation(tmp_path: Path) -> None:
    chapter = load_generation_chapter()
    tokenizer = CharacterTokenizer.from_text("春眠不觉晓")
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_sequence_length=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        dropout=0.0
    )
    model = MiniGPT(config)
    checkpoint_path = tmp_path / "mini-gpt-test.pth"

    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "training_steps": 0,
            "final_loss": 0.0
        },
        checkpoint_path
    )

    restored_model, restored_tokenizer, _ = (
        chapter.load_model_and_tokenizer(
            checkpoint_path=checkpoint_path,
            device=torch.device("cpu")
        )
    )

    generated_text = chapter.generate_text(
        model=restored_model,
        tokenizer=restored_tokenizer,
        prompt="春眠",
        max_new_tokens=3,
        strategy="greedy"
    )

    assert generated_text.startswith("春眠")
    assert len(generated_text) == 5
