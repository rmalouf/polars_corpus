"""Shared helpers for the test suite."""

import polars as pl
from polars_corpus import Match, SearchResults


def search_results(
    df: pl.DataFrame,
    query: str,
    matches: list[Match],
    variables: list[str] | None = None,
    **kwargs,
) -> SearchResults:
    """Results built from hand-written matches rather than by searching.

    A search names the variables its query binds, in binding order; matches
    written out here have only themselves to go on, so unless `variables`
    says otherwise they are named alphabetically.
    """
    if variables is None:
        variables = sorted({name for m in matches for name in m.bindings})
    return SearchResults(df, query, matches, variables, **kwargs)


def corpus(**columns: str) -> pl.DataFrame:
    """Build a corpus from whitespace-separated strings, one per column.

    >>> corpus(token="the dog", pos="DT NN").shape
    (2, 2)
    """
    return pl.DataFrame({name: value.split() for name, value in columns.items()})


def spans(results: SearchResults | None) -> list[tuple[int, int]]:
    """Flatten a search's matches to (start, end) tuples, None meaning none."""
    return [(m.span.start, m.span.end) for m in (results.matches if results else [])]


def jaccard(f12: pl.Expr, f1: pl.Expr, f2: pl.Expr, n: pl.Expr) -> pl.Expr:
    """A measure the library does not ship."""
    return f12 / (f1 + f2 - f12)


def named_by_alias(f12: pl.Expr, f1: pl.Expr, f2: pl.Expr, n: pl.Expr) -> pl.Expr:
    """The same measure, naming its own column instead of taking the def's name."""
    return jaccard(f12, f1, f2, n).alias("Jaccard")
