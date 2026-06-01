#!/usr/bin/env python3
"""Per-embedding-model similarity calibration profiles.

Cosine-similarity thresholds are NOT portable across embedding models. Each
model has its own distribution of pairwise cosine for unrelated text, so a
threshold that is "high" for one model can be "everything" for another.

Concrete failure this layer prevents: cashew's sleep cross-link/dedup and the
novelty gate were tuned for all-MiniLM-L6-v2 (unrelated-pair cosine mean
~0.13). After the brain migrated to thenlper/gte-large (mean ~0.765), the old
0.70 cross-link threshold matched 96% of ALL node pairs and saturated the
graph with ~15.6M edges in a single sleep run.

This module is the single source of truth for those constants. Every
similarity threshold in the codebase resolves through ``get_active_profile()``
so that switching ``DEFAULT_EMBEDDING_MODEL`` automatically switches the
constants. An unknown model raises ``UncalibratedModelError`` rather than
silently reusing another model's numbers: the constants must be measured (see
``scripts/calibrate_model.py``) and added here before that model can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelProfile:
    """Calibrated similarity constants for one embedding model.

    All thresholds are cosine similarities in [0, 1]. They must be measured
    against the model's actual unrelated-pair distribution, not guessed.
    """

    name: str
    dim: int
    cross_link_threshold: float  # cosine >= this and < dedup -> cross-link edge
    dedup_threshold: float       # cosine >= this -> dedup/merge candidate
    novelty_threshold: float     # reject a new node if nearest-neighbor cosine >= this
    notes: str = ""

    def __post_init__(self) -> None:
        for field in ("cross_link_threshold", "dedup_threshold", "novelty_threshold"):
            v = getattr(self, field)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{self.name}: {field}={v} not in [0, 1]")
        if not self.cross_link_threshold < self.dedup_threshold:
            raise ValueError(
                f"{self.name}: cross_link_threshold ({self.cross_link_threshold}) "
                f"must be < dedup_threshold ({self.dedup_threshold})"
            )


class UncalibratedModelError(RuntimeError):
    """Raised when an embedding model has no measured similarity profile."""


# Keyed by the canonical model name. Values were measured against real graphs;
# the `notes` field records the observed unrelated-pair distribution so the
# thresholds can be re-derived. Do NOT copy numbers between models.
MODEL_PROFILES: Dict[str, ModelProfile] = {
    "thenlper/gte-large": ModelProfile(
        name="thenlper/gte-large",
        dim=1024,
        cross_link_threshold=0.90,
        dedup_threshold=0.94,
        novelty_threshold=0.95,
        notes=(
            "Measured 2026-06-01 on a 4212-node graph (40k random pairs): "
            "unrelated-pair cosine mean 0.765, P95 0.831, P99 0.862. "
            "Per-node nearest-neighbor median 0.913, so novelty must sit above "
            "that to avoid rejecting distinct nodes; only true near-identicals "
            "exceed 0.95."
        ),
    ),
    "all-MiniLM-L6-v2": ModelProfile(
        name="all-MiniLM-L6-v2",
        dim=384,
        cross_link_threshold=0.70,
        dedup_threshold=0.82,
        novelty_threshold=0.82,
        notes=(
            "Historical cashew defaults. Unrelated-pair cosine mean ~0.13, "
            "P99 ~0.49, true dupes peak ~0.85-0.90."
        ),
    ),
}

# Aliases that map to a canonical profile key.
_ALIASES = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
}


def _resolve_key(model_name: str) -> Optional[str]:
    if model_name in MODEL_PROFILES:
        return model_name
    if model_name in _ALIASES:
        return _ALIASES[model_name]
    return None


def get_profile(model_name: str) -> ModelProfile:
    """Return the calibrated profile for *model_name*.

    Raises UncalibratedModelError if the model has no measured profile, so a
    new embedding model cannot be used until its thresholds are set.
    """
    key = _resolve_key(model_name)
    if key is None:
        raise UncalibratedModelError(
            f"No calibrated similarity profile for embedding model {model_name!r}. "
            f"Similarity thresholds are model-specific and must be measured before "
            f"use: run `python scripts/calibrate_model.py --db <graph.db>` and add a "
            f"ModelProfile entry to core/model_profiles.py. Known models: "
            f"{sorted(MODEL_PROFILES)}."
        )
    return MODEL_PROFILES[key]


def get_active_profile(model_name: Optional[str] = None) -> ModelProfile:
    """Return the profile for the active embedding model.

    If *model_name* is None, falls back to the configured DEFAULT_EMBEDDING_MODEL.
    """
    if model_name is None:
        from .config import DEFAULT_EMBEDDING_MODEL
        model_name = DEFAULT_EMBEDDING_MODEL
    return get_profile(model_name)
