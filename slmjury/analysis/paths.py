"""Where analysis scripts find their inputs and write their outputs.

Everything lives under one results folder, ``$SLMJURY_RESULTS`` if set, else ./results. ``core`` and
``experiment`` raise ``MissingResults`` with the expected path, so a missing input fails at once.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "ENV_VAR", "CORE_NAMES", "MissingResults",
    "results_dir", "default_dir", "core", "experiment", "analysis_dir",
    "analysis_figures_dir", "cache_dir", "require",
]

ENV_VAR = "SLMJURY_RESULTS"

# Folders under results/core/, written by the scripts/ runners.
CORE_NAMES = (
    "student_solutions", "judgements", "evaluations", "summaries",
    "personas", "juries", "debate", "scoring", "scoring_evaluations",
    "mtbench_responses", "mtbench_references",
)


class MissingResults(FileNotFoundError):
    """An expected results file or folder does not exist."""


def results_dir() -> Path:
    """The results root: ``$SLMJURY_RESULTS`` if set, else ``./results``."""
    return Path(os.environ.get(ENV_VAR) or "results")


def default_dir(*parts: str) -> str:
    """A folder under the results root, as a string for a runner's CLI default."""
    return str(results_dir().joinpath(*parts))


def require(path: Path, what: str = "input") -> Path:
    """Return ``path`` if it exists, else raise ``MissingResults``."""
    if not path.exists():
        raise MissingResults(
            f"{what} not found at {path}. Run from the repository root, or set "
            f"{ENV_VAR} to the folder that holds the results."
        )
    return path


def core(name: str) -> Path:
    """``results/core/<name>`` for one of the folders in ``CORE_NAMES``."""
    if name not in CORE_NAMES:
        raise ValueError(f"unknown core results folder {name!r}; expected one of {CORE_NAMES}")
    return require(results_dir() / "core" / name, f"results/core/{name}")


def experiment(name: str) -> Path:
    """``results/<name>``, the raw runs of one experiment."""
    return require(results_dir() / name, f"results/{name}")


def _created(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def analysis_dir(name: str) -> Path:
    """``results/analysis/<name>``, created if it does not exist."""
    return _created(results_dir() / "analysis" / name)


def analysis_figures_dir(name: str) -> Path:
    """``results/figures/analysis/<name>``, created if it does not exist."""
    return _created(results_dir() / "figures" / "analysis" / name)


def cache_dir(name: str) -> Path:
    """``results/cache/<name>``, created if it does not exist."""
    return _created(results_dir() / "cache" / name)
