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


def test_get_active_profile_follows_configured_model(monkeypatch):
    """get_active_profile() with no arg must resolve the CONFIGURED model, not a
    hardcoded default — otherwise every calibrated threshold silently uses the
    wrong model's numbers under a CASHEW_EMBEDDING_MODEL / config.yaml override."""
    from core import config as C
    monkeypatch.setattr(C, "get_embedding_model", lambda: "all-MiniLM-L6-v2")
    assert get_active_profile().name == "all-MiniLM-L6-v2"
    monkeypatch.setattr(C, "get_embedding_model", lambda: "thenlper/gte-large")
    assert get_active_profile().name == "thenlper/gte-large"


def test_profiles_have_calibrated_tension_band():
    """The tension band is model-specific: it must sit above the model's
    unrelated-pair mass and below its dedup level, or tension detection either
    matches everything or (the bug we fixed) nothing."""
    gte = get_profile("thenlper/gte-large")
    assert gte.tension_band == (0.83, 0.90)
    lo, hi = gte.tension_band
    assert 0.765 < lo < hi <= gte.dedup_threshold   # above unrelated mean, below dupes
    mini = get_profile("all-MiniLM-L6-v2")
    assert mini.tension_band == (0.30, 0.70)


def test_tension_band_invariant_enforced():
    with pytest.raises(ValueError):
        ModelProfile(name="bad", dim=8, cross_link_threshold=0.5,
                     dedup_threshold=0.6, novelty_threshold=0.7, tension_band=(0.7, 0.3))
