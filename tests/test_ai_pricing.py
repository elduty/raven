"""Tests for raven.ai.pricing — the fallback price table."""

import importlib

import pytest


def _reload_with(monkeypatch, value):
    """Reload the pricing module with RAVEN_AI_PRICES set to ``value``
    (or deleted when None) so the import-time parse runs against it."""
    if value is None:
        monkeypatch.delenv("RAVEN_AI_PRICES", raising=False)
    else:
        monkeypatch.setenv("RAVEN_AI_PRICES", value)
    import raven.ai.pricing as pricing
    return importlib.reload(pricing)


class TestLoadPrices:
    def test_unset_yields_empty(self, monkeypatch):
        pricing = _reload_with(monkeypatch, None)
        assert pricing._PRICES == {}

    def test_empty_string_yields_empty(self, monkeypatch):
        pricing = _reload_with(monkeypatch, "")
        assert pricing._PRICES == {}

    def test_invalid_json_yields_empty(self, monkeypatch):
        pricing = _reload_with(monkeypatch, "{not json")
        assert pricing._PRICES == {}

    def test_non_object_yields_empty(self, monkeypatch):
        pricing = _reload_with(monkeypatch, '["not", "a", "map"]')
        assert pricing._PRICES == {}

    def test_valid_table_parsed(self, monkeypatch):
        pricing = _reload_with(monkeypatch, '{"m":{"input":15,"output":75}}')
        assert pricing._PRICES == {"m": {"input": 15.0, "output": 75.0}}

    def test_entry_missing_rates_skipped(self, monkeypatch):
        pricing = _reload_with(
            monkeypatch,
            '{"good":{"input":1,"output":2},"bad":{"input":1}}',
        )
        assert "good" in pricing._PRICES
        assert "bad" not in pricing._PRICES


class TestCostUsd:
    def test_known_model_differential_in_out(self, monkeypatch):
        pricing = _reload_with(monkeypatch, '{"m":{"input":15,"output":75}}')
        # 1M input @ $15 + 1M output @ $75 = $90
        assert pricing.cost_usd("m", 1_000_000, 1_000_000) == pytest.approx(90.0)
        # output costs 5x input here — make sure they're not blended
        assert pricing.cost_usd("m", 1_000_000, 0) == pytest.approx(15.0)
        assert pricing.cost_usd("m", 0, 1_000_000) == pytest.approx(75.0)

    def test_unknown_model_returns_none_and_warns_once(self, monkeypatch, caplog):
        pricing = _reload_with(monkeypatch, '{"m":{"input":15,"output":75}}')
        with caplog.at_level("WARNING", logger="raven.ai.pricing"):
            assert pricing.cost_usd("other", 100, 100) is None
            assert pricing.cost_usd("other", 100, 100) is None  # again
        # warned exactly once for the unknown model
        warns = [r for r in caplog.records if "other" in r.message]
        assert len(warns) == 1

    def test_unset_table_returns_none(self, monkeypatch):
        pricing = _reload_with(monkeypatch, None)
        assert pricing.cost_usd("m", 100, 100) is None
