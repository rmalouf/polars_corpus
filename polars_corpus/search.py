from __future__ import annotations

import random
from typing import Optional

import polars as pl

from ._internal import py_concordance

__all__ = ["SearchResults", "kwic_concordance"]


class SearchResults:
    """Results of a search."""

    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matched_spans: list[tuple[int, int]],
    ) -> None:
        self._df = df
        self._query = query
        self._matched_spans = matched_spans

    def __repr__(self) -> str:
        return f"SearchResults<'{self._query}'; {len(self._matched_spans):,} matches>"

    def kwic_concordance(self, expr: pl.Expr, window_size: int = 5) -> pl.DataFrame:
        """Return a KWIC concordance dataframe.

        expr: Columns
        window_size: contex to include
        """

        return py_concordance(self._df.select(expr), self._matched_spans, window_size)

    def matches(self, expr: pl.Expr) -> pl.DataFrame:
        return py_concordance(self._df.select(expr), self._matched_spans, None)

    def head(self, n: int) -> SearchResults:
        if abs(n) > len(self._matched_spans):
            return self
        else:
            return SearchResults(self._df, self._query, self._matched_spans[:n])

    def tail(self, n: int) -> SearchResults:
        if n > len(self._matched_spans):
            return self
        elif n > 0:
            return SearchResults(self._df, self._query, self._matched_spans[-n:])
        else:
            raise ValueError

    def sample(self, k: int, seed: Optional[int] = None) -> SearchResults:
        state = random.getstate()
        random.seed(seed)
        if k < 0 or k > len(self._matched_spans):
            raise ValueError
        try:
            new_results = SearchResults(
                self._df, self._query, random.sample(self._matched_spans, k)
            )
        finally:
            random.setstate(state)
        return new_results

    # Do really want to do this? Am I assuming somewhere else that the spans are sorted?
    # We can always shuffle the concordances after it's built.
    def shuffle(self, seed: Optional[int] = None) -> SearchResults:
        state = random.getstate()
        random.seed(seed)
        try:
            new_results = SearchResults(
                self._df,
                self._query,
                random.sample(self._matched_spans, len(self._matched_spans)),
            )
        finally:
            random.setstate(state)
        return new_results


def kwic_concordance(
    search_results: SearchResults, expr: pl.Expr, window_size: int = 5
) -> pl.DataFrame:
    return search_results.kwic_concordance(expr, window_size)


def collocates(
    search_results: SearchResults, column: str, window_size: int = 5
) -> pl.DataFrame:
    f1 = search_results._df.lazy().group_by(column).len(name="f1")
    concordance = kwic_concordance(search_results, column, window_size).lazy()
    tbl = (
        concordance.select(
            context=pl.col(f"{column}_left_context")
            .list.concat(f"{column}_right_context")
            .explode()
        )
        .group_by("context")
        .len(name="f12")
        .join(f1, left_on="context", right_on="token", how="left")
        .with_columns(
            f2=pl.lit(concordance.height), n=search_results._df.height * window_size * 2
        )
    )
    return tbl.collect()
