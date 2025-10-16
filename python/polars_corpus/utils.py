import polars as pl

__all__ = ["ngrams"]


def ngrams(n: int, expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    exprs = [expr.alias(f"_0")] + [expr.shift(-i).alias(f"_{i}") for i in range(1, n)]
    return pl.struct(exprs)
