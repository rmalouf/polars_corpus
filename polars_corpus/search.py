from __future__ import annotations

import random
from typing import Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._internal import py_concordance, Span

__all__ = ["SearchResults", "concordance", "collocates"]


class SearchResults:
    """Results of a search."""

    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matched_spans: list[Span],
    ) -> None:
        self._df = df
        self._query = query
        self._matched_spans = matched_spans

    def __repr__(self) -> str:
        return f"SearchResults<'{self._query}'; {len(self._matched_spans):,} matches>"

    def concordance(self, expr: IntoExprColumn, context: str | int | tuple[int,int]) -> pl.DataFrame:
        """Return a KWIC concordance dataframe.

        expr: Columns
        window_size: contex to include
        """

        if isinstance(context, str):
            chunk_tag = self._df.get_column(context)
            left_window, right_window = 0, 0
        else:
            chunk_tag = None
            if isinstance(context, int):
                left_window = context
                right_window = context
            elif isinstance(context, tuple):
                left_window, right_window = context
            else:
                raise ValueError

        return py_concordance(
            self._df.select(expr),
            self._matched_spans,
            False,
            left_window,
            right_window,
            chunk_tag,
        )

    def matches(self, expr: pl.Expr) -> pl.DataFrame:
        return py_concordance(
            self._df.select(expr), self._matched_spans, True, 0, 0, None
        )

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


def concordance(search_results: SearchResults, expr: IntoExprColumn, context: str | int | tuple[int,int]) -> pl.DataFrame:
    return search_results.concordance(expr, context)


def collocates(
    search_results: SearchResults, column: str, window_size: int = 5
) -> pl.DataFrame:
    f1 = search_results._df.lazy().group_by(column).len(name="f1")
    conc = concordance(search_results, column, window_size)
    tbl = (
        conc.lazy()
        .select(
            collocate=pl.col(f"{column}_left_context")
            .list.concat(f"{column}_right_context")
            .explode()
        )
        .group_by("collocate")
        .len(name="f12")
        .join(f1, left_on="collocate", right_on="token", how="left")
        .with_columns(
            f2=pl.lit(conc.height), n=search_results._df.height * window_size * 2
        )
    )
    return tbl.collect()
