import re
from typing import Tuple, Dict, Any, Optional


def parse_api_request(s: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Parse 'API-Request: [ApiName(key1='value1', key2='value2')]' into (name, {key: value})
    """
    s = (s or "").strip()
    
    # Extract the part inside API-Request: [...]
    match = re.search(r'API-Request:\s*\[([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\]', s, re.DOTALL)
    if not match:
        # Try without API-Request: prefix
        match = re.search(r'\[([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\]', s, re.DOTALL)
    if not match:
        return None
    
    api_name = match.group(1)
    args_str = match.group(2).strip()
    
    # Parse key='value' or key="value" pairs
    args = {}
    # Match key='value' or key="value" patterns
    pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]([^'\"]*)['\"]"
    for m in re.finditer(pattern, args_str):
        args[m.group(1)] = m.group(2)
    
    return api_name, args


def semantic_match(pred: str, gold: str) -> bool:
    """
    Compare two API requests semantically:
    - Same API name
    - Same argument keys and values (order doesn't matter)
    """
    pred_parsed = parse_api_request(pred)
    gold_parsed = parse_api_request(gold)
    
    if pred_parsed is None or gold_parsed is None:
        # Fall back to normalized string comparison
        return normalize_string(pred) == normalize_string(gold)
    
    pred_name, pred_args = pred_parsed
    gold_name, gold_args = gold_parsed
    
    # Compare API name (case-insensitive)
    if pred_name.lower() != gold_name.lower():
        return False
    
    # Compare arguments
    if set(pred_args.keys()) != set(gold_args.keys()):
        return False
    
    for key in gold_args:
        if pred_args.get(key, "").strip() != gold_args[key].strip():
            return False
    
    return True


def normalize_string(s: str) -> str:
    """Normalize string for comparison: lowercase, remove extra whitespace."""
    s = (s or "").strip().lower()
    s = re.sub(r'\s+', ' ', s)  # Collapse whitespace
    s = s.replace('"', "'")     # Normalize quotes
    return s


def exact_match(pred: str, gold: str) -> bool:
    return (pred or "").strip() == (gold or "").strip()


def regex_match(pred: str, pattern: str) -> bool:
    try:
        return re.fullmatch(pattern, (pred or "").strip()) is not None
    except re.error:
        return False


def validate_answer(pred: str, gold: str, use_regex: bool = False) -> Tuple[bool, str]:
    # First try semantic matching for API requests
    if "API-Request:" in gold or "[" in gold:
        if semantic_match(pred, gold):
            return True, "semantic"
    
    if use_regex:
        ok = regex_match(pred, gold)
        return ok, "regex" if ok else "mismatch"
    
    ok = exact_match(pred, gold)
    return ok, "exact" if ok else "mismatch"


