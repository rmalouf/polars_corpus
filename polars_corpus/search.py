from __future__ import annotations

from collections import defaultdict

import polars as pl
import polars.selectors as cs

from ._internal import _to_spans, _make_spans_mask
from .cqp import matchall

__all__ = ["search", "SearchResults", "with_span_index", "to_span_index"]


class SearchResults:
    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matched_spans: list[tuple[int, int]],
        bindings: dict[str, list[str]],
    ) -> None:
        self.df = df
        self.query = query
        self.matched_spans = matched_spans
        self.bindings = bindings

    def __repr__(self) -> str:
        return f"SearchResults<'{self.query}'; {len(self.matched_spans):,} matches>"

    def matches(self, fields: str = "token") -> pl.DataFrame:
        columns = self.df.columns + ["var_" + v for v in self.bindings.keys()]

        df = (
            self.df.with_columns(self.to_spans(name="_spans"), *self.to_bindings())
            .pipe(with_span_index, span_col="_spans", name="_span_idx")
            .drop_nulls("_span_idx")
            .group_by("_span_idx")
            .agg(cs.all())
            .select(
                cs.by_name(columns).list.drop_nulls().list.unique(maintain_order=True)
            )
        )

        singleton_cols = [
            col
            for col in columns
            if df.select(pl.col(col).list.len().max()).item() == 1
        ]

        df = df.select(
            [
                pl.col(col).list.get(0) if col in singleton_cols else pl.col(col)
                for col in columns
            ]
        )

        return df

    def to_spans(self, name: str = "spans") -> pl.Series:
        try:
            return _to_spans(len(self.df), self.matched_spans).alias(name)
        except OverflowError:
            raise ValueError("negative index")

    def to_bindings(self, prefix: str = "var_") -> list[pl.Series]:
        try:
            n = len(self.df)
            cols = []
            for var, val in self.bindings.items():
                mask = _make_spans_mask(n, val)
                new_df = self.df.select(pl.col("token"), pl.Series("_mask", mask))
                new_col = new_df.select(
                    pl.when(pl.col("_mask"))
                    .then(pl.col("token"))
                    .otherwise(None)
                    .alias(prefix + var)
                ).get_column(prefix + var)
                cols.append(new_col)
            #            print(cols)
            return cols
        except OverflowError:
            raise ValueError("negative index")


def search(df: pl.DataFrame, query: str) -> SearchResults:
    spans, bindings = zip(*list(matchall(df, query)))
    new_bindings = defaultdict(list)
    for binding in bindings:
        for var, val in binding.items():
            new_bindings[var].append(val)
    return SearchResults(df, query, spans, new_bindings)


def to_span_index(
    spans: pl.Series, name: str = "span_idx", scheme: str = "BIO"
) -> pl.Series:
    if scheme != "BIO":
        raise NotImplementedError("Only BIO is supported")
    span_idx = pl.Series(spans == "B").cum_sum()
    span_idx = pl.when(spans == "O").then(pl.lit(None)).otherwise(span_idx)
    # span_idx = (spans != "O" & spans == "B").cum_sum()
    return span_idx.alias(name)


def with_span_index(
    df: pl.DataFrame, span_col: str, name: str = "span_idx", scheme: str = "BIO"
) -> pl.DataFrame:
    if scheme != "BIO":
        raise NotImplementedError("Only BIO is supported")
    return df.with_columns(to_span_index(df[span_col], name, scheme))


def with_spans(
    df: pl.DataFrame, matches: SearchResults, name: str = "spans"
) -> pl.DataFrame:
    return df.with_columns(matches.to_spans().alias(name))
