"""Shared helpers for the test suite."""

import polars as pl


def corpus(**columns: str) -> pl.DataFrame:
    """Build a corpus from whitespace-separated strings, one per column.

    >>> corpus(token="the dog", pos="DT NN")
    """
    return pl.DataFrame({name: value.split() for name, value in columns.items()})


def spans(matches) -> list[tuple[int, int]]:
    """Flatten matches (or None) to a list of (start, end) tuples."""
    return [(m.span.start, m.span.end) for m in (matches or [])]
