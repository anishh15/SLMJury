"""Answer extraction from model responses across dataset types.

Extracts final answers from model responses for numeric (GSM8K, GSM-Plus),
LaTeX (MATH), and multiple choice (ARC-Easy, ARC-Challenge) datasets.
Handles edge cases like repetitive outputs, malformed delimiters, and units.
"""

import re
from typing import Optional


def extract_answer(response: str, dataset_type: str = "numeric") -> str:
    """Extract the final answer from a model response based on dataset type.

    Args:
        response: The model's text response.
        dataset_type: One of "numeric", "latex", "multiple_choice".

    Returns:
        Extracted answer string, or "UNABLE_TO_EXTRACT" if not found.
    """
    if not response or not response.strip():
        return "UNABLE_TO_EXTRACT"

    extractors = {
        "multiple_choice": _extract_multiple_choice,
        "latex": _extract_latex_answer,
    }
    return extractors.get(dataset_type, _extract_numeric_answer)(response)


def get_dataset_type(dataset_name: str) -> str:
    """Map a dataset name to its answer type.

    Args:
        dataset_name: Dataset name (e.g., "gsm8k", "math", "arc_easy").

    Returns:
        One of "numeric", "latex", or "multiple_choice".
    """
    name = dataset_name.lower().replace("-", "_").replace(" ", "_")

    if name in ("gsm8k", "gsm_plus", "gsmplus"):
        return "numeric"
    elif name in ("math", "hendrycks_math", "competition_math"):
        return "latex"
    elif name in (
        "arc_easy", "arc_challenge", "arc", "ai2_arc",
        "hellaswag", "winogrande", "truthfulqa", "truthful_qa",
    ):
        return "multiple_choice"
    return "numeric"


# --- Multiple choice extraction (ARC-Easy, ARC-Challenge) ---

def _extract_multiple_choice(response: str) -> str:
    """Extract multiple choice answer (A-D or 1-4) from response."""
    if not response:
        return "UNABLE_TO_EXTRACT"

    text = response.strip()

    # Priority 1: \boxed{X}
    boxed = re.search(r'\\boxed\s*\{\s*([A-Da-d1-4])\s*\}', text)
    if boxed:
        return boxed.group(1).upper()

    # Priority 2: Explicit answer patterns (search from end)
    patterns = [
        r'(?:the\s+)?answer\s+is[:\s]*\(?([A-Da-d1-4])\)?',
        r'(?:the\s+)?correct\s+answer\s+is[:\s]*\(?([A-Da-d1-4])\)?',
        r'answer[:\s]+\(?([A-Da-d1-4])\)?',
        r'\b([A-Da-d])\)\s+is\s+(?:the\s+)?(?:correct|right)\b',
        r'(?:option|choice)\s+\(?([A-Da-d1-4])\)?\s+is\s+(?:correct|right)',
        r'(?:select|choose)\s+\(?([A-Da-d1-4])\)?',
        r'^\s*\(?([A-Da-d])\)?[\.:\)]?\s*$',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            result = matches[-1].group(1).upper()
            return result if not result.isdigit() else result

    # Priority 3: Standalone letter at end
    end_match = re.search(r'\b([A-Da-d])\s*[\.)]?\s*$', text)
    if end_match:
        return end_match.group(1).upper()

    # Priority 4: Letter with context clues
    option_match = re.search(r'\b([A-Da-d])\)\s', text)
    if option_match:
        last_letter = None
        for m in re.finditer(
            r'\b([A-Da-d])\)?(?:\s+is|\s*[\.，]?\s*(?:because|since|as))',
            text, re.IGNORECASE
        ):
            last_letter = m.group(1).upper()
        if last_letter:
            return last_letter

    # Priority 5: Any standalone letter A-D
    all_letters = re.findall(r'\b([A-Da-d])\b', text)
    if all_letters:
        return all_letters[-1].upper()

    # Priority 6: Raw number 1-4
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
    stripped = response.strip()
    if stripped in num_to_letter:
        return num_to_letter[stripped]

    num_match = re.search(r'\b([1-4])\s*$', text)
    if num_match:
        return num_to_letter[num_match.group(1)]

    return "UNABLE_TO_EXTRACT"


# --- LaTeX answer extraction (MATH dataset) ---

def _extract_latex_answer(response: str) -> str:
    """Extract answer from LaTeX \\boxed{} format with nested brace handling."""
    if not response:
        return "UNABLE_TO_EXTRACT"

    # Try \boxed{...} with proper brace matching
    boxed = _extract_last_boxed(response)
    if boxed:
        return boxed.strip()

    # Fallback 1: #### delimiter
    if "####" in response:
        numeric = _extract_from_delimiter(response)
        if numeric:
            return numeric

    # Fallback 2: Natural language answer patterns
    for pattern in [
        r'(?:final\s+answer|the\s+answer\s+is)[:\s]*([^\n\.]+)',
        r'(?:therefore|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n\.]+)',
    ]:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # Fallback 3: Last math expression
    for pattern in [r'=\s*([^\n=]+)$', r'(-?\d+\.?\d*)\s*$']:
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

    return "UNABLE_TO_EXTRACT"


def _extract_last_boxed(text: str) -> Optional[str]:
    """Extract content from the last \\boxed{...} with nested brace handling."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        idx = text.rfind("boxed{")
        if idx == -1:
            return None
        start = idx + len("boxed{")
    else:
        start = idx + len("\\boxed{")

    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == "{":
            brace_count += 1
        elif text[end] == "}":
            brace_count -= 1
        end += 1

    if brace_count != 0:
        return None
    return text[start:end - 1]


# --- Numeric answer extraction (GSM8K, GSM-Plus) ---

def _extract_numeric_answer(response: str) -> str:
    """Extract final numeric answer using multiple fallback strategies."""
    if not response or not response.strip():
        return "UNABLE_TO_EXTRACT"

    # Strategy 1: #### delimiter (GSM8K standard)
    if "####" in response:
        answer = _extract_from_delimiter(response)
        if answer:
            return answer

    # Strategy 2: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    if boxed:
        cleaned = _clean_numeric(boxed[-1])
        if cleaned:
            return cleaned

    # Strategy 3: Natural language answer patterns
    for pattern in [
        r'(?:final\s+answer|the\s+answer\s+is)[:\s]*([^\n.]+)',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
        r'=\s*([^\n=]+)$',
    ]:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            cleaned = _clean_numeric(matches[-1])
            if cleaned:
                return cleaned

    # Strategy 4: Large comma-separated numbers
    comma_nums = re.findall(r'\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?', response)
    if comma_nums:
        cleaned = _clean_numeric(comma_nums[-1])
        if cleaned:
            return cleaned

    # Strategy 5: Last number in response
    all_nums = re.findall(r'-?\d+\.?\d*', response)
    if all_nums:
        return all_nums[-1]

    return "UNABLE_TO_EXTRACT"


def _extract_from_delimiter(response: str) -> Optional[str]:
    """Extract answer after #### delimiter, handling repetitive and malformed outputs."""
    normalized = re.sub(r'####\s*:', '####', response)
    parts = normalized.split("####")
    if len(parts) <= 1:
        return None

    candidates = []
    for part in parts[1:]:
        segment = part.strip()
        if not segment:
            continue

        first_line_match = re.match(r'^([^\n]*?)(?:\n\s*[a-zA-Z]|$)', segment)
        first_segment = (
            first_line_match.group(1).strip()
            if first_line_match
            else segment.split('\n')[0].strip()
        )

        immediate = _extract_immediate_answer(first_segment)
        if immediate:
            candidates.append(immediate)

    if not candidates:
        return None

    # Use voting for repetitive outputs
    counts: dict[str, int] = {}
    for ans in candidates:
        counts[ans] = counts.get(ans, 0) + 1
    return max(counts, key=counts.get)


def _extract_immediate_answer(segment: str) -> Optional[str]:
    """Extract the immediate numeric value from a segment after ####."""
    if not segment:
        return None

    match = re.match(
        r'^\$?(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:[a-z]+)?(?:\s|\.|$)',
        segment, re.IGNORECASE
    )
    if match:
        return _normalize_num_str(match.group(1))

    num_match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', segment)
    if num_match:
        return _normalize_num_str(num_match.group())

    return None


def _normalize_num_str(num_str: str) -> str:
    """Normalize a numeric string: remove commas, convert whole floats to int."""
    num_str = num_str.replace(',', '')
    try:
        value = float(num_str)
        return str(int(value)) if value == int(value) else num_str
    except ValueError:
        return num_str


def _clean_numeric(text: str) -> Optional[str]:
    """Clean and normalize a numeric string, handling units and multipliers."""
    if not text or not text.strip():
        return None

    text = text.strip()

    # Remove LaTeX artifacts
    text = re.sub(r'\\text\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = text.replace('{', '').replace('}', '')

    # Handle word multipliers
    multiplier = 1
    for word, mult in [
        ('trillion', 1_000_000_000_000),
        ('billion', 1_000_000_000),
        ('million', 1_000_000),
        ('thousand', 1_000),
    ]:
        if word in text.lower():
            multiplier = mult
            break

    # Strip symbols and units
    text = re.sub(r'[$£€%¥₹]', '', text)
    text = text.replace(',', '')
    text = re.sub(
        r'\s*(trillion|billion|million|thousand)s?\s*', ' ',
        text, flags=re.IGNORECASE
    )

    # Extract number with optional unit suffix
    match = re.search(
        r'(-?\d+\.?\d*)\s*'
        r'(?:dollars?|cents?|liters?|litres?|grams?|kilograms?|kg|'
        r'meters?|metres?|feet|foot|inches?|miles?|hours?|minutes?|'
        r'seconds?|days?|weeks?|months?|years?|people|persons?|items?|'
        r'pieces?|blocks?|points?|calories?|gallons?|oranges?|apples?|'
        r'cookies?|teachers?|students?)?\s*\.?\s*$',
        text, re.IGNORECASE
    )

    num_str = match.group(1) if match else None
    if not num_str:
        fallback = re.search(r'-?\d+\.?\d*', text)
        if not fallback:
            return None
        num_str = fallback.group()

    # Apply multiplier and format
    try:
        value = float(num_str) * multiplier
        return str(int(value)) if value == int(value) else str(value)
    except ValueError:
        return None
