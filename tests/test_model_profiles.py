"""Tests for the per-embedding-model similarity profile layer."""

import pytest

from core.model_profiles import (
    MODEL_PROFILES,
    ModelProfile,
    UncalibratedModelError,
    get_active_profile,
    get_profile,
)


def test_known_models_have_profiles():
    assert "thenlper/gte-large" in MODEL_PROFILES
    assert "all-MiniLM-L6-v2" in MODEL_PROFILES


def test_get_profile_returns_calibrated_values():
    p = get_profile("thenlper/gte-large")
    assert p.cross_link_threshold == 0.90
    assert p.dedup_threshold == 0.94
    assert p.novelty_threshold == 0.95


def test_minilm_keeps_historical_values():
    p = get_profile("all-MiniLM-L6-v2")
    assert p.cross_link_threshold == 0.70
    assert p.dedup_threshold == 0.82


def test_alias_resolves():
    p = get_profile("sentence-transformers/all-MiniLM-L6-v2")
    assert p.name == "all-MiniLM-L6-v2"


def test_unknown_model_raises():
    with pytest.raises(UncalibratedModelError):
        get_profile("some/unmeasured-model-v9")


def test_get_active_profile_uses_configured_default():
    # DEFAULT_EMBEDDING_MODEL is gte-large in config.
    p = get_active_profile()
    assert p.name == "thenlper/gte-large"


def test_profile_invariants_enforced():
    # cross_link must be strictly below dedup.
    with pytest.raises(ValueError):
        ModelProfile(
            name="bad", dim=8,
            cross_link_threshold=0.95, dedup_threshold=0.90, novelty_threshold=0.96,
        )
    # thresholds must be in [0, 1].
    with pytest.raises(ValueError):
        ModelProfile(
            name="bad2", dim=8,
            cross_link_threshold=0.5, dedup_threshold=0.6, novelty_threshold=1.5,
        )


def test_all_registered_profiles_are_valid():
    # Construction already validates, but assert the ordering invariant holds
    # for every shipped profile as a guard against future edits.
    for p in MODEL_PROFILES.values():
        assert 0.0 <= p.cross_link_threshold < p.dedup_threshold <= 1.0
        assert 0.0 <= p.novelty_threshold <= 1.0
