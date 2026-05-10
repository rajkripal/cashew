"""Tests for temporal-anchor preservation during sleep-cycle cluster merges.

Regression: cluster merge synthesis was dropping date/weekday/relative-time
phrases that appeared in source nodes, which crippled temporal-reasoning
accuracy on downstream benchmarks (LoCoMo cat-2 specifically).
"""
import pytest

from core.sleep import (
    SleepProtocol,
    _collect_temporal_anchors,
    _has_any_anchor,
)


class TestTemporalAnchorDetection:
    def test_iso_date(self):
        anchors = _collect_temporal_anchors(["met on 2026-03-05 at noon"])
        assert "2026-03-05" in anchors

    def test_month_day_year(self):
        anchors = _collect_temporal_anchors(["went on March 5, 2026"])
        assert any("march" in a for a in anchors)

    def test_weekday(self):
        anchors = _collect_temporal_anchors(["call last Tuesday went well"])
        assert any("tuesday" in a for a in anchors)
        # the relative phrase "last tuesday" should also be captured
        assert any("last tuesday" in a for a in anchors)

    def test_relative_time(self):
        anchors = _collect_temporal_anchors(["shipped two weeks ago"])
        assert any("two weeks ago" in a for a in anchors)

    def test_bare_year(self):
        anchors = _collect_temporal_anchors(["graduated in 2018"])
        assert "2018" in anchors

    def test_no_temporal_content(self):
        assert _collect_temporal_anchors(["raj likes ramen"]) == []

    def test_dedup_across_snippets(self):
        anchors = _collect_temporal_anchors([
            "trip in March 2026",
            "March 2026 was great",
        ])
        # "march 2026" should appear once, not twice
        assert sum(1 for a in anchors if "march 2026" in a) == 1

    def test_has_any_anchor_positive(self):
        assert _has_any_anchor("we met March 5", ["march 5"])

    def test_has_any_anchor_negative(self):
        assert not _has_any_anchor("we met somewhere", ["march 5", "tuesday"])

    def test_has_any_anchor_empty(self):
        assert not _has_any_anchor("anything", [])
        assert not _has_any_anchor("", ["march 5"])


class TestSynthesisPreservesAnchors:
    def _proto(self):
        # SleepProtocol needs only its method; pass an empty path.
        return SleepProtocol.__new__(SleepProtocol)

    def test_no_model_returns_longest(self):
        out = self._proto()._synthesize_cluster_content(
            ["short", "a much longer snippet here"], ["fact", "fact"], model_fn=None
        )
        assert out == "a much longer snippet here"

    def test_llm_keeps_anchor_passes_through(self):
        snippets = ["raj traveled to sweden in march 2026", "sweden trip in march 2026 was great"]
        def model_fn(_p):
            return "raj traveled to sweden in march 2026 and enjoyed it"
        out = self._proto()._synthesize_cluster_content(snippets, ["fact"]*2, model_fn=model_fn)
        assert "march 2026" in out.lower()

    def test_llm_drops_all_anchors_falls_back_to_longest(self):
        snippets = [
            "raj traveled to sweden in march 2026",
            "sweden trip in march 2026 was great",
        ]
        def model_fn(_p):
            # synthesis bleaches the date out
            return "raj traveled to sweden and enjoyed it greatly overall"
        out = self._proto()._synthesize_cluster_content(snippets, ["fact"]*2, model_fn=model_fn)
        # Falls back to the longest source; both contain "march 2026"
        assert "march 2026" in out.lower()

    def test_no_anchors_in_sources_no_fallback(self):
        # If sources have no temporal info, the LLM output should be accepted
        # as-is even though it has no anchor.
        snippets = ["raj likes ramen", "raj enjoys ramen a lot"]
        def model_fn(_p):
            return "raj enjoys ramen"
        out = self._proto()._synthesize_cluster_content(snippets, ["fact"]*2, model_fn=model_fn)
        assert out == "raj enjoys ramen"

    def test_short_llm_response_falls_back(self):
        # Existing behavior preserved: <10 char LLM responses fall through.
        snippets = ["raj traveled to sweden in march 2026", "sweden trip in march"]
        def model_fn(_p):
            return "ok"
        out = self._proto()._synthesize_cluster_content(snippets, ["fact"]*2, model_fn=model_fn)
        assert "march" in out.lower()
