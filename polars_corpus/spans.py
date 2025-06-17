from __future__ import annotations

import polars as pl

from ._internal import _with_spans

__all__ = ["with_span_index", "with_spans"]


def with_span_index(
    df: pl.DataFrame, span_col: str, name: str = "span_idx", scheme: str = "BIO"
) -> pl.DataFrame:
    if scheme != "BIO":
        raise NotImplementedError("Only BIO is supported")
    span_idx = pl.Series(df[span_col] == "B").cum_sum()
    span_idx = pl.when(df[span_col] == "O").then(pl.lit(None)).otherwise(span_idx)
    return df.with_columns(span_idx.alias(name))


def with_spans(df: pl.DataFrame, concordance, name: str = "spans") -> pl.TPolarsFrame:
    try:
        return df.with_columns(_with_spans(len(df), concordance).alias(name))
    except OverflowError:
        raise ValueError('negative index')


#     spans = df.select(pl.repeat(pl.lit("O"), pl.count()).alias(name)).get_column(name)
#     #starts = (s.start for s in concordance)
#     starts =  [s.start for s in concordance]
#     spans = spans.scatter(starts, "B")
#     #ranges = chain.from_iterable(range(s.start + 1, s.end) for s in concordance)
#     ranges = [i for s in concordance for i in range(s.start + 1, s.end)]
#     spans = spans.scatter(ranges, "I")
#     return df.with_columns(spans)


# def with_spans(df: TPolarsFrame, concordance, name: str = "spans") -> pl.TPolarsFrame:
#     spans = df.select(pl.repeat(pl.lit("O"), pl.count()).alias(name)).get_column(name)
#     #starts = (s.start for s in concordance)
#     starts =  [s.start for s in concordance]
#     spans = spans.scatter(starts, "B")
#     #ranges = chain.from_iterable(range(s.start + 1, s.end) for s in concordance)
#     ranges = [i for s in concordance for i in range(s.start + 1, s.end)]
#     spans = spans.scatter(ranges, "I")
#     return df.with_columns(spans)

# LAZY VERSION
# def with_spans(df: TPolarsFrame, concordance, name: str = "spans") -> pl.TPolarsFrame:
#     df = df.lazy()
#
#     starts = [s.start for s in concordance]
#     middles = list(chain.from_iterable(range(s.start + 1, s.end) for s in concordance))
#
#     return df.with_row_index().with_columns(
#         pl.when(pl.col('index').is_in(starts))
#         .then(pl.lit("B"))
#         .when(pl.col('index').is_in(middles))
#         .then(pl.lit("I"))
#         .otherwise(pl.lit("O"))
#         .alias(name)).drop("index").collect()
