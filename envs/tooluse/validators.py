import re
from typing import Tuple


def exact_match(pred: str, gold: str) -> bool:
    return (pred or "").strip() == (gold or "").strip()


def regex_match(pred: str, pattern: str) -> bool:
    try:
        return re.fullmatch(pattern, (pred or "").strip()) is not None
    except re.error:
        return False


def validate_answer(pred: str, gold: str, use_regex: bool = False) -> Tuple[bool, str]:
    if use_regex:
        ok = regex_match(pred, gold)
        return ok, "regex" if ok else "mismatch"
    ok = exact_match(pred, gold)
    return ok, "exact" if ok else "mismatch"


