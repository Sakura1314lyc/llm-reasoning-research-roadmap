from fractions import Fraction

from reward_functions import accuracy_reward, extract_answer, format_reward, length_penalty


def test_strict_and_fallback_extraction() -> None:
    assert extract_answer("steps... #### 1,200") == Fraction(1200)
    assert extract_answer("final value 3/4") == Fraction(3, 4)
    assert extract_answer("no answer") is None


def test_equivalent_fraction_and_decimal() -> None:
    assert accuracy_reward("#### 0.5", "1/2") == 1.0


def test_format_reward_requires_ending() -> None:
    assert format_reward("#### 6") == 1.0
    assert format_reward("#### 6\nextra") == 0.0


def test_wrong_or_missing_answer() -> None:
    assert accuracy_reward("#### 5", "6") == 0.0
    assert accuracy_reward("unknown", "6") == 0.0


def test_length_penalty() -> None:
    assert length_penalty("x" * 20, maximum_characters=20) == -1.0
