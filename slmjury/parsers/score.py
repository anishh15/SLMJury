"""Score parsing (1-5) for open-ended SLM judge responses.

Single scores, MT-Bench style (\\boxed{4}), and multi-dimension scores, SummEval style
(\\boxed{coherence=4, consistency=3, ...}), each through a priority cascade like judgement.py's.
"""

import re
from typing import Optional


# Dimension names for SummEval multi-score extraction
SUMMEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")

# Longest digit run handed to int(). Scores are 1..max_score, so three digits
# is generous. A degenerate output can hold thousands of digits, and Python
# refuses int() on strings over 4,300 digits; such a run is skipped and the
# score is unparseable (None), which every caller already handles.
_MAX_SCORE_DIGITS = 3


def parse_score(response: str, max_score: int = 5) -> Optional[int]:
    """Extract one score from a judge response: an integer in [1, max_score], or None.

    Tries, in order: \\boxed{N} or \\boxed{N/5}, **N** or **N/5**, [[N]], "Score: N" / "Rating: N" /
    "N out of 5", then the last standalone integer in [1, max_score].
    """
    if not response or not response.strip():
        return None

    text = response.strip()

    # Method 1: \boxed{N} or \boxed{N/5}
    boxed_matches = re.findall(
        r'\\*boxed\s*\{\s*(\d+)\s*(?:/\s*\d+)?\s*\}', text,
    )
    if boxed_matches and len(boxed_matches[-1]) <= _MAX_SCORE_DIGITS:
        score = int(boxed_matches[-1])  # last match wins
        if 1 <= score <= max_score:
            return score

    # Method 2: **N** or **N/5**
    bold_matches = re.findall(r'\*\*\s*(\d+)\s*(?:/\s*\d+)?\s*\*\*', text)
    if bold_matches and len(bold_matches[-1]) <= _MAX_SCORE_DIGITS:
        score = int(bold_matches[-1])
        if 1 <= score <= max_score:
            return score

    # Method 3: [[N]]
    bracket_matches = re.findall(r'\[\[\s*(\d+)\s*\]\]', text)
    if bracket_matches and len(bracket_matches[-1]) <= _MAX_SCORE_DIGITS:
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
        if matches and len(matches[-1]) <= _MAX_SCORE_DIGITS:
            score = int(matches[-1])
            if 1 <= score <= max_score:
                return score

    # Method 5: last standalone integer in valid range
    standalone = re.findall(r'\b(\d+)\b', text)
    for s in reversed(standalone):
        # No score in 1..max_score is this long (see _MAX_SCORE_DIGITS).
        if len(s) > _MAX_SCORE_DIGITS:
            continue
        score = int(s)
        if 1 <= score <= max_score:
            return score

    return None


def _extract_boxed_content(text: str) -> Optional[str]:
    """Content of \\boxed{...} with nested braces handled, or None if there is no box.

    Counts brace depth, since [^}]* stops early on \\boxed{\\text{coherence}=2, ...}.
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
    """Extract per-dimension scores from a SummEval-style response; returns {dimension: int or None}.

    Tries, in order: structured boxed (\\boxed{coherence=4, ...}, also inside \\text{} or $...$), positional boxed
    (\\boxed{3,2,2,3}, in dimension order), per-dimension boxed ("Coherence: \\boxed{4}"), then labels ("Coherence: 4").
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
            if dim_match and len(dim_match.group(1)) <= _MAX_SCORE_DIGITS:
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
                if len(num_str) > _MAX_SCORE_DIGITS:
                    all_valid = False
                    continue
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
            if matches and len(matches[-1]) <= _MAX_SCORE_DIGITS:
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
            if matches and len(matches[-1]) <= _MAX_SCORE_DIGITS:
                score = int(matches[-1])
                if 1 <= score <= max_score:
                    result[dim] = score
                    break

    return result
