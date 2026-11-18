"""Read the letter ("A" or "B") a pairwise judge committed to.

Reasoning judges explain first, some decline to choose, and the article "a" is not a vote, so ``read_choice``
accepts only a committed choice. Anything else is "Undefined" and is left out of the flip rate.
"""

from __future__ import annotations

import re

__all__ = ["read_choice"]

_BARE = re.compile(r"[\s*_`\"'(\[]*([AB])[\s*_`\"')\].:!]*")
_LETTER = re.compile(r"(?<![A-Za-z])([AB])(?![A-Za-z])")
_BOXED = re.compile(r"\\boxed\{\s*(?:\\text\{\s*)?([AB])\s*\}?\s*\}")
_HEDGE = re.compile(r"(?<![A-Za-z])[AB]\s*(?:(?i:and|or)\b|&|/)")
_CONCLUDE = [
    re.compile(r"(?i:answer|response|option)\s+\**\(?([AB])\)?\**\s+(?i:is\s+(?:the\s+)?(?:more\s+correct|better|more\s+accurate|preferable|correct))"),
    re.compile(r"(?i:(?:correct|best|better|right|final)\s+(?:answer|choice|option)\s*(?:is|:)|verdict\s*(?:is|:)|decision\s*(?:is|:)|my\s+(?:answer|choice)\s*(?:is|:)|the\s+answer\s+is)[\s:*]*(?:(?i:answer|option)\s+)?\(?\**([AB])\b"),
    re.compile(r"(?i:i'd|i\s+would|i\s+will|i'll|i)\s+(?i:pick|choose|select|go\s+with)\s+(?:(?i:answer)\s+)?\**([AB])\b"),
    re.compile(r"\*\*\s*(?:(?i:answer)\s+)?([AB])\s*\*\*"),
]


def _body(text):
    # GPT-OSS writes its reasoning and its answer to separate channels; the
    # answer follows "assistantfinal". Reasoning judges may leave a think block.
    if "assistantfinal" in text:
        text = text.split("assistantfinal")[-1]
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def read_choice(text):
    """The letter ("A" or "B") a pairwise judge committed to, or "Undefined"."""
    t = _body(text or "")
    # A response that is a bare letter (markdown and punctuation allowed).
    if _BARE.fullmatch(t):
        return _BARE.fullmatch(t).group(1)
    # A boxed letter, if every boxed letter agrees.
    boxed = set(_BOXED.findall(t))
    if boxed:
        return boxed.pop() if len(boxed) == 1 else "Undefined"
    lines = [l for l in t.splitlines() if l.strip()]
    if not lines:
        return "Undefined"
    # A bare letter on the first or the last line, if the two do not disagree.
    first, last = _BARE.fullmatch(lines[0]), _BARE.fullmatch(lines[-1])
    if first or last:
        picks = {m.group(1) for m in (first, last) if m}
        return picks.pop() if len(picks) == 1 else "Undefined"
    # A conclusion in the last sentence ("the final answer is A") that names one letter and does not hedge.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", lines[-1].strip()) if s]
    final = sentences[-1] if sentences else ""
    if len(set(_LETTER.findall(final))) != 1 or _HEDGE.search(final):
        return "Undefined"
    for pat in _CONCLUDE:
        m = pat.search(final)
        if m:
            return m.group(1)
    return "Undefined"
