## productivity metrics

import polars as pl
from polars._typing import IntoExprColumn

__all__ = [
    "frequency_spectrum",
    "yules_K",
    "count_hapaxes",
]


def frequency_spectrum(expr: IntoExprColumn, sort=False) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    spectrum = (
        expr.value_counts(name="m", parallel=True)
        .struct.field("m")
        .value_counts(name="V(m,N)", parallel=True)
    )
    if sort:
        spectrum = spectrum.sort()
    return spectrum


def count_hapaxes(expr: IntoExprColumn) -> pl.Expr:
    return expr.is_unique().sum()


def yules_K(expr: IntoExprColumn) -> pl.Expr:
    spectrum = frequency_spectrum(expr)
    N = expr.len()
    return (
        10000
        * (
            (spectrum.struct.field("m").pow(2) * spectrum.struct.field("V(m,N)") - N)
            / N**2
        ).sum()
    )
