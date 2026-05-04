"""Score parsing for open-ended SLM judge responses.

Extracts numerical scores (1-5) from judge model outputs for:
- Single scores: MT-Bench style (\\boxed{4})
- Multi-dimension scores: SummEval style (\\boxed{coherence=4, consistency=3, ...})

Uses a priority cascade similar to judgement.py but for numerical scores.
"""

import re
from typing import Optional


# Dimension names for SummEval multi-score extraction
SUMMEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")


def parse_score(response: str, max_score: int = 5) -> Optional[int]:
    """Extract a single numerical score from a judge response.

    Priority cascade:
    1. \\boxed{N} or \\boxed{N/5}
    2. **N** or **N/5** (markdown bold)
    3. [[N]] (bracket format)
    4. "Score: N" / "Rating: N" / "N out of 5" patterns
    5. Last standalone integer in [1, max_score]

    Args:
        response: The judge model's response text.
        max_score: Maximum valid score (default: 5).

    Returns:
        Integer score in [1, max_score], or None if unable to extract.
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Method 1: \boxed{N} or \boxed{N/5}
    boxed_matches = re.findall(
        r'\\*boxed\s*\{\s*(\d+)\s*(?:/\s*\d+)?\s*\}', text,
    )
    if boxed_matches:
        score = int(boxed_matches[-1])  # last match wins
        if 1 <= score <= max_score:
            return score

    # Method 2: **N** or **N/5**
    bold_matches = re.findall(r'\*\*\s*(\d+)\s*(?:/\s*\d+)?\s*\*\*', text)
    if bold_matches:
        score = int(bold_matches[-1])
        if 1 <= score <= max_score:
            return score

    # Method 3: [[N]]
    bracket_matches = re.findall(r'\[\[\s*(\d+)\s*\]\]', text)
    if bracket_matches:
        score = int(bracket_matches[-1])
        if 1 <= score <= max_score:
            return score

    # Method 4: labeled patterns
    labeled_patterns = [
        r'(?:score|rating|grade)\s*[:\-=]\s*(\d+)\s*(?:/\s*\d+)?',
        r'(\d+)\s*(?:out of|/)\s*' + str(max_score),
        r'(?:overall|final)\s*(?:score|rating)\s*[:\-=]?\s*(\d+)',
    ]
    for pattern in labeled_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            score = int(matches[-1])
            if 1 <= score <= max_score:
                return score

    # Method 5: last standalone integer in valid range
    standalone = re.findall(r'\b(\d+)\b', text)
    for s in reversed(standalone):
        score = int(s)
        if 1 <= score <= max_score:
            return score

    return None


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...} handling nested braces.

    Standard [^}]* fails on \\boxed{\\text{coherence}=2, ...} because
    \\text{} has nested braces. This function counts brace depth.

    Returns:
        The inner content string, or None if no boxed found.
    """
    # Find last occurrence of \boxed{ (last match wins, consistent with parse_score)
    pattern = r'\\+boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None

    # Use last match
    match = matches[-1]
    start = match.end()  # position after the opening {
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth == 0:
        return text[start:i - 1]  # exclude the closing }
    return None


def parse_multi_score(
    response: str,
    dimensions: tuple[str, ...] = SUMMEVAL_DIMENSIONS,
    max_score: int = 5,
) -> dict[str, Optional[int]]:
    """Extract multi-dimension scores from a SummEval-style judge response.

    Handles all observed model output patterns via priority cascade:
    1. Structured boxed: \\boxed{coherence=4, consistency=3, ...}
       (also handles \\text{} wrapping and $\\boxed{...}$ math mode)
    2. Positional boxed: \\boxed{3,2,2,3} (scores in dimension order)
    3. Per-dimension scattered boxed: "Coherence: \\boxed{4}" or
       "\\boxed{coherence=4}" appearing individually in text
    4. Per-dimension labeled: "Coherence: 4" / "**Coherence**: 4"

    Args:
        response: The judge model's response text.
        dimensions: Tuple of dimension names to extract.
        max_score: Maximum valid score (default: 5).

    Returns:
        Dict mapping dimension name → int score or None.
    """
    result = {dim: None for dim in dimensions}

    if not response or not response.strip():
        return result

    text = response.strip()

    # ── Method 1: Structured boxed with nested brace support ──
    # Handles: \boxed{coherence=4, consistency=3, fluency=5, relevance=4}
    #          \boxed{\text{coherence}=2, \text{consistency}=5, ...}
    #          $\boxed{coherence=2, consistency=5, ...}$
    boxed_content = _extract_boxed_content(text)
    if boxed_content:
        # Strip \text{} wrappers: \text{coherence} -> coherence
        cleaned = re.sub(r'\\text\s*\{([^}]*)\}', r'\1', boxed_content)
        for dim in dimensions:
            dim_match = re.search(
                rf'{dim}\s*[=:]\s*(\d+)', cleaned, re.IGNORECASE,
            )
            if dim_match:
                score = int(dim_match.group(1))
                if 1 <= score <= max_score:
                    result[dim] = score

        # Return early only if ALL dimensions were found
        if all(v is not None for v in result.values()):
            return result

    # ── Method 2: Positional comma-separated boxed ──
    # Handles: \boxed{3,2,2,3} — scores in dimension order
    if boxed_content and all(v is None for v in result.values()):
        nums = re.findall(r'\d+', boxed_content)
        if len(nums) == len(dimensions):
            all_valid = True
            for dim, num_str in zip(dimensions, nums):
                score = int(num_str)
                if 1 <= score <= max_score:
                    result[dim] = score
                else:
                    all_valid = False
            if all_valid:
                return result
            result = {dim: None for dim in dimensions}

    # ── Method 3: Per-dimension scattered boxed ──
    # Handles: "Coherence: \boxed{4}" / "- Coherence: $\boxed{4}$"
    #          "Coherence: \boxed{coherence=4}"
    #          "**Coherence**: \boxed{4}"
    for dim in dimensions:
        patterns = [
            # "Coherence: \boxed{coherence=4}" or "\boxed{coherence=4}"
            rf'\\+boxed\s*\{{\s*{dim}\s*[=:]\s*(\d+)\s*(?:/\s*\d+)?\s*\}}',
            # "Coherence: \boxed{4}" or "Coherence: $\boxed{4}$"
            rf'{dim}[^\\]{{0,50}}\\+boxed\s*\{{\s*(\d+)\s*(?:/\s*\d+)?\s*\}}',
            # "**Coherence** ... \boxed{4}" (bold heading then boxed)
            rf'\*\*{dim}\*\*[^\\]{{0,80}}\\+boxed\s*\{{\s*(\d+)\s*(?:/\s*\d+)?\s*\}}',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                score = int(matches[-1])
                if 1 <= score <= max_score:
                    result[dim] = score
                    break

    if any(v is not None for v in result.values()):
        return result

    # ── Method 4: Per-dimension labeled patterns ──
    # Handles: "Coherence: 4" / "Coherence = 4" / "Coherence: 4/5"
    #          "**Coherence**: 4" / "Coherence (4/5)" / "Coherence - 4"
    for dim in dimensions:
        patterns = [
            rf'\*\*{dim}\*\*\s*[:\-=]\s*(\d+)\s*(?:/\s*\d+)?',
            rf'{dim}\s*[:\-=]\s*(\d+)\s*(?:/\s*\d+)?',
            rf'{dim}\s*\(\s*(\d+)\s*/\s*\d+\s*\)',
            rf'{dim}\s*(?:score|rating)?\s*[:\-=]\s*(\d+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                score = int(matches[-1])
                if 1 <= score <= max_score:
                    result[dim] = score
                    break

    return result
