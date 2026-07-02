"""Tests for the temporal-anchor detection helpers (_collect_temporal_anchors,
_has_any_anchor) used to recognise date/weekday/relative-time phrases."""
import pytest

from core.sleep import _collect_temporal_anchors, _has_any_anchor


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
