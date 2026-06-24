"""Discount calculation for the billing module."""


def apply_discount(price: float, percent: float) -> float:
    """Apply a percentage discount to ``price`` and round to 2 decimals.

    ``percent`` must be in the inclusive range 0..100; anything outside
    raises ``ValueError``.
    """
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    discounted = price * (1 - percent / 100)
    return round(discounted, 2)
