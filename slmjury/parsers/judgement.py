"""Judgement parsing for SLM judge responses.

Extracts verdicts (CORRECT/INCORRECT) from judge model responses using a
priority cascade: boxed format → markdown bold → labeled → quick verdict →
sentence patterns → fallback. Each method has an associated confidence level.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ParseResult:
    """Result of parsing a judge response."""

    verdict: str  # "Correct", "Incorrect", or "Undefined"
    confidence: str  # "high", "medium", "low", "none"
    method: str  # Which extraction method succeeded
    raw_match: Optional[str]  # The actual matched text


def parse_judgement(response: str) -> str:
    """Parse judge response to extract verdict.

    Args:
        response: The judge model's response text.

    Returns:
        "Correct", "Incorrect", or "Undefined".
    """
    return parse_judgement_detailed(response).verdict


def parse_judgement_detailed(response: str) -> ParseResult:
    """Parse judge response with detailed extraction information.

    Tries extraction methods in priority order:
    1. \\boxed{} format (high confidence)
    2. **bold** markdown (high confidence)
    3. Labeled patterns like "VERDICT: X" (medium confidence)
    4. Quick verdict — standalone word at start (medium confidence)
    5. Sentence patterns like "The answer is X" (medium confidence)
    6. Last occurrence of verdict word (low confidence)

    Args:
        response: The judge model's response text.

    Returns:
        ParseResult with verdict, confidence, method, and raw match.
    """
    if not response or not response.strip():
        return ParseResult("Undefined", "none", "empty_response", None)

    text = response.strip()
    text_lower = text.lower()

    # Priority cascade
    extractors = [
        ("high", "boxed", _extract_boxed_verdict),
        ("high", "markdown_bold", _extract_markdown_bold_verdict),
        ("medium", "labeled", _extract_labeled_verdict),
        ("medium", "quick_verdict", lambda t: _extract_quick_verdict(text)),
        ("medium", "sentence_pattern", _extract_sentence_verdict),
        ("low", "fallback", _extract_fallback_verdict),
    ]

    for confidence, method, extractor in extractors:
        result = extractor(text_lower)
        if result:
            return ParseResult(result[0], confidence, method, result[1])

    return ParseResult("Undefined", "none", "no_match", None)


def get_verdict_confidence(response: str) -> Tuple[str, str]:
    """Get verdict with confidence level (legacy interface).

    Args:
        response: The judge model's response text.

    Returns:
        Tuple of (verdict, confidence).
    """
    result = parse_judgement_detailed(response)
    return result.verdict, result.confidence


def parse_batch(responses: List[str]) -> List[ParseResult]:
    """Parse multiple judge responses.

    Args:
        responses: List of judge response strings.

    Returns:
        List of ParseResult objects.
    """
    return [parse_judgement_detailed(r) for r in responses]


def get_parsing_stats(results: List[ParseResult]) -> dict:
    """Compute statistics over a batch of parse results.

    Args:
        results: List of ParseResult from parse_batch.

    Returns:
        Dictionary with counts and rates by verdict, confidence, and method.
    """
    total = len(results)
    if total == 0:
        return {}

    correct = sum(1 for r in results if r.verdict == "Correct")
    incorrect = sum(1 for r in results if r.verdict == "Incorrect")
    undefined = sum(1 for r in results if r.verdict == "Undefined")

    methods: dict[str, int] = {}
    for r in results:
        methods[r.method] = methods.get(r.method, 0) + 1

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "undefined": undefined,
        "defined_rate": (correct + incorrect) / total,
        "confidence": {
            level: sum(1 for r in results if r.confidence == level)
            for level in ("high", "medium", "low", "none")
        },
        "methods": methods,
    }


# --- Extraction methods (ordered by priority) ---

def _extract_boxed_verdict(text_lower: str) -> Optional[Tuple[str, str]]:
    """Extract verdict from \\boxed{} format (last match wins)."""
    matches = []
    for match in re.finditer(
        r'\\*boxed\s*\{\s*([^}]*(?:incorrect|correct)[^}]*)\s*\}',
        text_lower, re.IGNORECASE
    ):
        content = match.group(1).strip().lower()
        if 'incorrect' in content:
            matches.append((match.start(), "Incorrect", match.group(0)))
        elif 'correct' in content:
            matches.append((match.start(), "Correct", match.group(0)))

    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return (matches[-1][1], matches[-1][2])


def _extract_markdown_bold_verdict(text_lower: str) -> Optional[Tuple[str, str]]:
    """Extract verdict from **CORRECT** or **INCORRECT** markdown bold."""
    matches = []
    for pattern, verdict in [
        (r'\*\*\s*incorrect\s*\*\*', "Incorrect"),
        (r'\*\*\s*correct\s*\*\*', "Correct"),
    ]:
        for match in re.finditer(pattern, text_lower):
            matches.append((match.start(), verdict, match.group(0)))

    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return (matches[-1][1], matches[-1][2])


def _extract_labeled_verdict(text_lower: str) -> Optional[Tuple[str, str]]:
    """Extract verdict from labels like 'VERDICT: CORRECT' or 'The answer is INCORRECT'."""
    patterns = [
        r'(?:verdict|answer|final|judgement|judgment|conclusion)\s*[:\-=]\s*(incorrect|correct)',
        r'(?:the\s+)?(?:verdict|answer|final\s+answer|judgement|judgment)\s+is\s*[:\-=]?\s*(incorrect|correct)',
        r'(?:my\s+)?(?:verdict|answer|judgement|judgment)\s*[:\-=]\s*(incorrect|correct)',
    ]

    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            verdict = "Correct" if match.group(1).lower() == "correct" else "Incorrect"
            matches.append((match.start(), verdict, match.group(0)))

    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return (matches[-1][1], matches[-1][2])


def _extract_quick_verdict(response: str) -> Optional[Tuple[str, str]]:
    """Extract quick verdict from responses where first word is 'Correct'/'Incorrect'."""
    stripped = response.strip()
    first_line = stripped.split('\n')[0].strip()
    first_word = stripped.split()[0] if stripped.split() else ""

    for text in (stripped, first_line, first_word):
        check = text.lower().rstrip('.,!?:;')
        if check == "correct" or check.startswith("correct"):
            return ("Correct" if not check.startswith("incorrect") else "Incorrect", text)
        if check.startswith("incorrect"):
            return ("Incorrect", text)

    return None


def _extract_sentence_verdict(text_lower: str) -> Optional[Tuple[str, str]]:
    """Extract verdict from sentence patterns like 'The answer is correct'."""
    patterns = [
        r'(?:the\s+)?(?:answer|solution|response|result)\s+is\s+(incorrect|correct)',
        r"student(?:'s|s')?\s+(?:answer|solution|response)\s+is\s+(incorrect|correct)",
        r'\bthis\s+is\s+(incorrect|correct)',
        r'(?:i\s+)?(?:judge|determine|find|conclude|assess)\s+(?:this|it|the answer)\s+(?:as\s+|to be\s+)?(incorrect|correct)',
        r'(?:therefore|thus|hence|so)[,\s]+(incorrect|correct)',
        r'my\s+(?:verdict|answer|conclusion)[:\s]+(incorrect|correct)',
    ]

    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            verdict = "Correct" if match.group(1).lower() == "correct" else "Incorrect"
            matches.append((match.start(), verdict, match.group(0)))

    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return (matches[-1][1], matches[-1][2])


def _extract_fallback_verdict(text_lower: str) -> Optional[Tuple[str, str]]:
    """Fallback: find last occurrence of 'correct'/'incorrect' in text."""
    correct_matches = list(re.finditer(r'(?<!in)correct', text_lower))
    incorrect_matches = list(re.finditer(r'incorrect', text_lower))

    last_correct = correct_matches[-1].start() if correct_matches else -1
    last_incorrect = incorrect_matches[-1].start() if incorrect_matches else -1

    if last_incorrect >= 0 and last_correct >= 0:
        if last_incorrect > last_correct:
            return ("Incorrect", "incorrect")
        return ("Correct", "correct")
    elif last_incorrect >= 0:
        return ("Incorrect", "incorrect")
    elif last_correct >= 0:
        return ("Correct", "correct")

    return None
