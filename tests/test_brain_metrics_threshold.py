"""Guard: brain-metrics' near-duplicate count must use the configured model's
dedup threshold, not a hardcoded 0.82 (MiniLM's). On gte-large a 0.82 cutoff
counted ~10% of ALL node pairs as near-duplicates (639,906 vs 4 at the
calibrated 0.94). brain-metrics.py is hyphenated (not importable), so this
guards the source directly."""
import os


def test_brain_metrics_near_dup_uses_profile_threshold():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "brain-metrics.py")
    src = open(path).read()
    assert "get_active_profile().dedup_threshold" in src, "near-dup threshold must come from the profile"
    assert "sim > 0.82" not in src, "the hardcoded MiniLM 0.82 threshold must be gone"
