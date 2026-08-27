"""可复用且可测试的数学答案与格式奖励。"""

from fractions import Fraction
import re


STRICT = re.compile(r"####\s*([^\n]+?)\s*$")
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def normalize_number(value: str) -> Fraction | None:
    match = NUMBER.search(value.replace(",", ""))
    if match is None:
        return None
    try:
        return Fraction(match.group(0))
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text: str, strict: bool = False) -> Fraction | None:
    match = STRICT.search(text.strip())
    if match:
        return normalize_number(match.group(1))
    if strict:
        return None
    numbers = NUMBER.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def accuracy_reward(text: str, ground_truth: str) -> float:
    prediction = extract_answer(text)
    target = normalize_number(ground_truth)
    return 1.0 if prediction is not None and prediction == target else 0.0


def format_reward(text: str) -> float:
    return 1.0 if extract_answer(text, strict=True) is not None else 0.0


def length_penalty(text: str, maximum_characters: int = 2000) -> float:
    return -1.0 if len(text) >= maximum_characters else 0.0


def total_reward(text: str, ground_truth: str) -> float:
    return accuracy_reward(text, ground_truth) + 0.1 * format_reward(text) + length_penalty(text)
