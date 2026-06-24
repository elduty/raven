"""Small text helpers."""


def slugify(value: str) -> str:
    """Lowercase a string and replace runs of whitespace with single hyphens.

    Leading/trailing whitespace is stripped first so the result never
    starts or ends with a hyphen.
    """
    return "-".join(value.lower().split())


def truncate(value: str, limit: int) -> str:
    """Return at most ``limit`` characters of ``value``."""
    if limit < 0:
        raise ValueError("limit must be non-negative")
    return value[:limit]
