"""Pinned Parquet downloads from the Hugging Face Hub, and the size check every loaded dataset passes."""

from __future__ import annotations


class DatasetLoadError(RuntimeError):
    """A dataset failed to load in full; raised before any inference starts."""


def read_parquet_rows(repo_id: str, filename: str, revision: str) -> list[dict]:
    """Rows of one Parquet file in a Hub dataset repository, at a pinned commit."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo_id, filename, repo_type="dataset", revision=revision)
    except Exception as e:
        raise DatasetLoadError(
            f"could not download {filename} from {repo_id} at {revision}: {e}") from e
    return pq.read_table(path).to_pylist()


def check_size(name: str, items: list, expected: int | None = None, source: str = "") -> list:
    """Return ``items``, or raise DatasetLoadError if they are empty or not ``expected`` long."""
    where = f" from {source}" if source else ""
    if not items:
        raise DatasetLoadError(f"{name}: no items loaded{where}. Stopping before any inference.")
    if expected is not None and len(items) != expected:
        raise DatasetLoadError(
            f"{name}: {len(items)} items loaded{where}, but the study uses {expected}. "
            "Stopping before any inference.")
    return items
