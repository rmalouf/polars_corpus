from __future__ import annotations

import polars as pl
import polars.selectors as cs

from ._internal import _to_spans
from .cqp import cqp

__all__ = ["search", "SearchResults", "with_span_index", "to_span_index"]


class SearchResults:
    def __init__(
        self, df: pl.DataFrame, query: str, matched_spans: list[tuple[int, int]]
    ) -> None:
        self.df = df
        self.query = query
        self.matched_spans = matched_spans

    def __repr__(self) -> str:
        return f"SearchResults<'{self.query}', {len(self.matched_spans):,} matches>"

    def matches(self, fields: str = "token") -> pl.DataFrame:
        columns = self.df.columns

        df = (
            self.df.with_columns(self.to_spans(name="_spans"))
            .pipe(with_span_index, span_col="_spans")
            .drop_nulls()
            .group_by("span_idx")
            .agg(cs.all())
            .select(cs.by_name(columns).list.unique(maintain_order=True))
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
            return pl.Series(_to_spans(len(self.df), self.matched_spans).alias(name))
        except OverflowError:
            raise ValueError("negative index")


def search(df: pl.DataFrame, query: str) -> SearchResults:
    parsed_query = cqp.parse_string(query, parse_all=True)[0]
    return SearchResults(df, query, list(parsed_query.matchall(df)))


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
