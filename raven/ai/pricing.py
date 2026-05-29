"""pricing.py — fallback AI cost estimation from a configurable price table.

This is the *fallback* cost source. The preferred source is the per-call cost
the backend reports (Claude CLI ``total_cost_usd``; LiteLLM
``x-litellm-response-cost``). This table is consulted only when the backend
reports nothing — i.e. plain OpenAI-compatible endpoints (vLLM, raw OpenAI
without a cost-reporting proxy).

Configured via the ``RAVEN_AI_PRICES`` env var: JSON mapping model name → USD
per 1,000,000 tokens, with separate input/output rates::

    RAVEN_AI_PRICES={"some-model":{"input":15,"output":75}}

Unset / empty / invalid → empty table (no fallback cost). The reviewer treats
a ``None`` return as "no price known" and records zero cost + a one-time
warning, so a gap is visible in logs rather than silently wrong.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _load_prices() -> dict[str, dict[str, float]]:
    """Parse ``RAVEN_AI_PRICES`` once at import. Degrades to {} on any problem
    (unset, empty, invalid JSON, wrong shape) with a single startup warning —
    never crash-loops the container over a cost-estimation knob."""
    raw = (os.environ.get("RAVEN_AI_PRICES") or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("RAVEN_AI_PRICES is not valid JSON (%s) — fallback cost table disabled", e)
        return {}
    if not isinstance(data, dict):
        logger.warning("RAVEN_AI_PRICES must be a JSON object {model: {input, output}} — disabled")
        return {}
    table: dict[str, dict[str, float]] = {}
    for model, rates in data.items():
        if not isinstance(rates, dict) or "input" not in rates or "output" not in rates:
            logger.warning("RAVEN_AI_PRICES[%r] must have numeric 'input' and 'output' — skipping", model)
            continue
        try:
            table[model] = {"input": float(rates["input"]), "output": float(rates["output"])}
        except (TypeError, ValueError):
            logger.warning("RAVEN_AI_PRICES[%r] input/output not numeric — skipping", model)
    return table


_PRICES = _load_prices()

# Models we've already warned about (missing from the table), so the log isn't
# spammed once per review. Bounded by the number of distinct models.
_warned_models: set[str] = set()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost in USD from the configured price table.

    Returns ``None`` when the model has no entry (caller distinguishes
    "no price known" from a genuine $0.00) and warns once per unknown model.
    """
    rates = _PRICES.get(model)
    if rates is None:
        if model not in _warned_models:
            _warned_models.add(model)
            logger.warning(
                "No price-table entry for model %r and no provider-reported cost — "
                "recording 0 cost. Set RAVEN_AI_PRICES to estimate it.", model,
            )
        return None
    return input_tokens / 1_000_000 * rates["input"] + output_tokens / 1_000_000 * rates["output"]
