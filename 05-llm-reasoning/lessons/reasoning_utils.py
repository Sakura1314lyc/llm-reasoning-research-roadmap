"""推理实验共用的答案抽取、归一化和投票函数。"""

from collections import Counter
from fractions import Fraction
import re


STRICT_PATTERN = re.compile(r"####\s*([^\n]+?)\s*$")
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def extract_answer(text: str) -> str | None:
    """优先抽取 #### 结尾，其次使用最后一个数字。"""
    strict = STRICT_PATTERN.search(text.strip())
    candidate = strict.group(1) if strict else None
    if candidate is None:
        numbers = NUMBER_PATTERN.findall(text)
        candidate = numbers[-1] if numbers else None
    return normalize_answer(candidate) if candidate is not None else None


def normalize_answer(answer: str) -> str | None:
    value = answer.strip().replace(",", "").replace("$", "")
    match = NUMBER_PATTERN.search(value)
    if match is None:
        return None
    token = match.group(0)
    try:
        number = Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None
    return str(number.numerator) if number.denominator == 1 else f"{number.numerator}/{number.denominator}"


def majority_vote(outputs: list[str]) -> tuple[str | None, Counter]:
    answers = [answer for output in outputs if (answer := extract_answer(output)) is not None]
    counts = Counter(answers)
    return (counts.most_common(1)[0][0] if counts else None), counts


def exact_match(prediction: str | None, reference: str) -> bool:
    return prediction is not None and prediction == normalize_answer(reference)
