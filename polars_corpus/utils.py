import polars as pl

from ._typing import T_Frame

__all__ = ["with_chunk_index", "ngrams"]


def with_chunk_index(df: T_Frame, column: str, name: str = "chunk_idx") -> T_Frame:
    return (
        df.with_columns(pl.col(column).eq("B").alias(name))
        .with_columns(pl.col(name).cum_sum())
        .with_columns(
            pl.when(pl.col(column).eq("O"))
            .then(None)
            .otherwise(pl.col(name))
            .alias(name)
        )
    )


def ngrams(n: int, expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    exprs = [expr.alias(f"_0")] + [expr.shift(-i).alias(f"_{i}") for i in range(1, n)]
    return pl.struct(exprs)
