import polars as pl
from polars._typing import IntoExprColumn

__all__ = ["ngrams", "output_name"]


def output_name(expr: IntoExprColumn) -> str:
    """Name of the column produced by `expr`.

    Parameters
    ----------
    expr : IntoExprColumn
        A column name, a Polars expression, or a Series.

    Returns
    -------
    str
        The name the column carries once `expr` is evaluated.
    """
    if isinstance(expr, str):
        return expr
    if isinstance(expr, pl.Series):
        return expr.name
    return expr.meta.output_name()


def ngrams(n: int, expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    exprs = [expr.alias("_0")] + [expr.shift(-i).alias(f"_{i}") for i in range(1, n)]
    return pl.struct(exprs)
