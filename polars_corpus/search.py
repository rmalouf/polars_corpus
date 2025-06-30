from __future__ import annotations

import random
from typing import Optional

import polars as pl

from ._internal import py_concordance

__all__ = ["SearchResults", "kwic_concordance"]


class SearchResults:
    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matched_spans: list[tuple[int, int]],
    ) -> None:
        self.df = df
        self.query = query
        self.matched_spans = matched_spans

    def __repr__(self) -> str:
        return f"SearchResults<'{self.query}'; {len(self.matched_spans):,} matches>"

    def kwic_concordance(self, expr: pl.Expr, window_size: int = 5) -> pl.DataFrame:
        return py_concordance(self.df.select(expr), self.matched_spans, window_size)

    def matches(self, expr: pl.Expr) -> pl.DataFrame:
        return py_concordance(self.df.select(expr), self.matched_spans, None)

    def head(self, n: int) -> SearchResults:
        if abs(n) > len(self.matched_spans):
            return self
        else:
            return SearchResults(self.df, self.query, self.matched_spans[:n])

    def tail(self, n: int) -> SearchResults:
        if n > len(self.matched_spans):
            return self
        elif n > 0:
            return SearchResults(self.df, self.query, self.matched_spans[-n:])
        else:
            raise ValueError

    def sample(self, k: int, seed: Optional[int] = None) -> SearchResults:
        state = random.getstate()
        random.seed(seed)
        if k < 0 or k > len(self.matched_spans):
            raise ValueError
        try:
            new_results = SearchResults(
                self.df, self.query, random.sample(self.matched_spans, k)
            )
        finally:
            random.setstate(state)
        return new_results

    def shuffle(self, seed: Optional[int] = None) -> SearchResults:
        state = random.getstate()
        random.seed(seed)
        try:
            new_results = SearchResults(
                self.df,
                self.query,
                random.sample(self.matched_spans, len(self.matched_spans)),
            )
        finally:
            random.setstate(state)
        return new_results


def kwic_concordance(
    search_results: SearchResults, expr: pl.Expr, window_size: int = 5
) -> pl.DataFrame:
    return search_results.kwic_concordance(expr, window_size)

    #
    # def to_chunks(self, name: str = "chunks") -> pl.Series:
    #     try:
    #         return _to_chunks(len(self.df), self.matched_spans).alias(name)
    #     except OverflowError:
    #         raise ValueError("negative index")
    #
    # def to_bindings(self, prefix: str = "var_") -> list[pl.Series]:
    #     try:
    #         n = len(self.df)
    #         cols = []
    #         for var, val in self.bindings.items():
    #             mask = _make_spans_mask(n, val)
    #             new_df = self.df.select(pl.col("token"), pl.Series("_mask", mask))
    #             new_col = new_df.select(
    #                 pl.when(pl.col("_mask"))
    #                 .then(pl.col("token"))
    #                 .otherwise(None)
    #                 .alias(prefix + var)
    #             ).get_column(prefix + var)
    #             cols.append(new_col)
    #         #            print(cols)
    #         return cols
    #     except OverflowError:
    #         raise ValueError("negative index")


# def to_chunk_index(
#     spans: pl.Series, name: str = "span_idx", scheme: str = "BIO"
# ) -> pl.Series:
#     if scheme != "BIO":
#         raise NotImplementedError("Only BIO is supported")
#     span_idx = pl.Series(spans == "B").cum_sum()
#     span_idx = pl.when(spans == "O").then(pl.lit(None)).otherwise(span_idx)
#     # span_idx = (spans != "O" & spans == "B").cum_sum()
#     return span_idx.alias(name)


# def with_chunk_index(
#     df: pl.DataFrame, span_col: str, name: str = "span_idx", scheme: str = "BIO"
# ) -> pl.DataFrame:
#     if scheme != "BIO":
#         raise NotImplementedError("Only BIO is supported")
#     return df.with_columns(
#         (pl.col(span_col) == "B").cum_sum().alias(name)
#     ).with_columns(
#         pl.when(pl.col(span_col) == "O")
#         .then(pl.lit(None))
#         .otherwise(pl.col("_span_idx"))
#         .alias(name)
#     )
#
#
# def with_chunks(
#     df: pl.DataFrame, matches: SearchResults, name: str = "chunks"
# ) -> pl.DataFrame:
#     return df.with_columns(matches.to_chunks().alias(name))
