from typing import TypeVar, overload

import polars as pl

all = ["pmi"]

T = TypeVar("T", bound=pl.DataFrame | pl.LazyFrame)


@overload
def tabulate(df: pl.DataFrame, x: str, y: str) -> pl.DataFrame: ...


@overload
def tabulate(df: pl.LazyFrame, x: str, y: str) -> pl.LazyFrame: ...


def tabulate(
    df: pl.DataFrame | pl.LazyFrame, x: str, y: str
) -> pl.DataFrame | pl.LazyFrame:
    t = (
        df.select(x, y)
        .group_by(x, y)
        .len(f"f_{x}_{y}")
        .with_columns(
            pl.col(f"f_{x}_{y}").sum().over(x).alias(f"f_{x}"),
            pl.col(f"f_{x}_{y}").sum().over(y).alias(f"f_{y}"),
        )
    )
    return t.select(x, y, f"f_{x}_{y}", f"f_{x}", f"f_{y}")


def pmi(df: pl.DataFrame, x: pl.Expr, y: pl.Expr) -> pl.DataFrame:
    t = (
        df.select(x.alias("x"), y.alias())
        .group_by("x", "y")
        .len("f_xy")
        .with_columns(
            f_x=pl.col("f_xy").sum().over("x"), f_y=pl.col("f_xy").sum().over("y")
        )
    )
    n = t["f_xy"].sum()
    t = t.with_columns(
        pmi=((pl.col("f_xy") * n) / (pl.col("f_x") * pl.col("f_y"))).log()
    )
    return t.select("x", "y", "f12", "f1", "f2", "pmi")
