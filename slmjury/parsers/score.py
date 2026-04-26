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


def parse_multi_score(
    response: str,
    dimensions: tuple[str, ...] = SUMMEVAL_DIMENSIONS,
    max_score: int = 5,
) -> dict[str, Optional[int]]:
    """Extract multi-dimension scores from a SummEval-style judge response.

    Priority cascade:
    1. \\boxed{coherence=4, consistency=3, fluency=5, relevance=4}
    2. Per-dimension patterns: "coherence: 4" / "Coherence = 4" / "Coherence: 4/5"

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

    # Method 1: structured boxed format
    # Match \boxed{coherence=4, consistency=3, fluency=5, relevance=4}
    boxed_match = re.search(
        r'\\*boxed\s*\{([^}]*)\}', text, re.IGNORECASE,
    )
    if boxed_match:
        content = boxed_match.group(1)
        for dim in dimensions:
            dim_match = re.search(
                rf'{dim}\s*[=:]\s*(\d+)', content, re.IGNORECASE,
            )
            if dim_match:
                score = int(dim_match.group(1))
                if 1 <= score <= max_score:
                    result[dim] = score

        # If we found at least some dimensions, return
        if any(v is not None for v in result.values()):
            return result

    # Method 2: per-dimension labeled patterns scattered in text
    for dim in dimensions:
        patterns = [
            rf'{dim}\s*[:\-=]\s*(\d+)\s*(?:/\s*\d+)?',
            rf'{dim}\s*(?:score|rating)?\s*[:\-=]\s*(\d+)',
            rf'\*\*{dim}\*\*\s*[:\-=]\s*(\d+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                score = int(matches[-1])
                if 1 <= score <= max_score:
                    result[dim] = score
                    break

    return result
