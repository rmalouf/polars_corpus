from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from ._typing import T_Frame

__all__ = [
    "crosstab",
    "compute_mi",
    "compute_min_sens",
    "compute_loglik",
    "assoc",
]
LIB = Path(__file__).parent

def crosstab(df: T_Frame, x: str, y: str) -> T_Frame:
    """
    Creates a crosstabulation (contingency table) from a given dataframe with two variables.

    The crosstab includes frequencies of occurrence as well as marginal and total sums.

    Args:
        df: The input data as a Polars DataFrame or LazyFrame.
        x: The column name of the first variable (independent variable).
        y: The column name of the second variable (dependent variable).

    Returns:
        A Polars DataFrame containing the contingency table with columns:
        - x: Levels of the first variable
        - y: Levels of the second variable
        - f12: Joint frequencies
        - f1: Row marginal sums
        - f2: Column marginal sums
        - n: Grand total

    Raises:
        ValueError: If x or y columns don't exist in the dataframe
    """
    t = (
        df.select(x, y)
        .drop_nulls([x, y])
        .group_by(x, y)
        .agg(pl.len().alias("f12"))
        .with_columns(
            [
                pl.col("f12").sum().over(x).alias("f1"),
                pl.col("f12").sum().over(y).alias("f2"),
                pl.col("f12").sum().alias("n"),
            ]
        )
    )

    return t


def _validated_crosstab(df: T_Frame) -> T_Frame:
    # required_cols = ["f12", "n", "f1", "f2"]
    # if not all(col in df.columns for col in required_cols):
    #    raise ValueError(f"Missing required columns. Expected: {required_cols}")
    return df.filter(
        pl.col("f1") > 0, pl.col("f2") > 0, pl.col("f12") >= 0, pl.col("n") > 0
    )


def compute_mi(table: T_Frame) -> T_Frame:
    """
    Computes the Pointwise Mutual Information (PMI) for the given table using its columns.

    :param table: Input table containing the necessary columns to compute PMI. The table should
        include at least four columns: `f12` (joint frequency of two events), `n` (total
        number of observations), `f1` (frequency of the first event), and `f2` (frequency of
        the second event). The columns are expected to use these exact names.
    :type table: T_Frame
    :return: The same input table with an additional column `pmi` that contains the computed
        PMI values for each row.
    :rtype: T_Frame
    """
    return _validated_crosstab(table).with_columns(
        pmi=((pl.col("f12") * pl.col("n")) / (pl.col("f1") * pl.col("f2"))).log()
    )


def loglik(
    f12: IntoExprColumn, f1: IntoExprColumn, f2: IntoExprColumn, n: IntoExprColumn
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[f12, f1, f2, n],
        function_name="loglik",
        is_elementwise=True,
    )


def compute_loglik(table: T_Frame) -> T_Frame:
    table = _validated_crosstab(table)
    data = loglik(table["f12"], table["f1"], table["f2"], table["n"]).alias("LL")
    return table.with_columns(data)


def compute_min_sens(table: T_Frame) -> T_Frame:
    """
    Compute the mean square (ms) values for a given data table. The function
    calculates the minimum horizontal value between two derived columns,
    "f12 / f1" and "f12 / f2", and appends it as a new column named `ms` to
    the original data table.

    :param table: The input table represented as a T_Frame. The table is
        expected to have columns named "f12", "f1", and "f2", with numeric
        values involved in the calculations.
    :return: A T_Frame instance with an additional column `ms` containing
        the computed mean square values.
    """
    return table.with_columns(
        min_sens=pl.min_horizontal(
            pl.col("f12") / pl.col("f1"), pl.col("f12") / pl.col("f2")
        )
    )


def assoc(df: T_Frame, x: str, y: str, method: str, min_freq: int = 0) -> T_Frame:
    """
    Calculate association metrics between two categorical variables in a dataframe.

    This function computes specific association metrics between two categorical
    variables ('x' and 'y') from a given dataframe. It supports the calculation
    of either Pointwise Mutual Information (MI) or Mutual Strength (MS) based
    on the provided method. The function considers a minimum frequency threshold
    to filter the contingency table before computing the desired metric.

    :param df: A dataframe containing the categorical variables to analyze.
    :param x: The name of the first categorical variable.
    :param y: The name of the second categorical variable.
    :param method: The method to compute association metrics. Accepted values are
        'pmi' for Pointwise Mutual Information and 'ms' for Mutual Strength.
    :param min_freq: The minimum frequency threshold to filter the contingency
        table. Default is 0.
    :return: A dataframe containing the computed association metrics.
    :raises ValueError: If an unknown method is specified.
    """
    table: T_Frame = crosstab(df, x, y).filter(pl.col("f12") >= min_freq)
    match method:
        case "mi":
            return compute_mi(table)
        case "min_sens":
            return compute_min_sens(table)
        case "loglik":
            return compute_loglik(table)
        case _:
            raise ValueError(f"Unknown method: {method}")
