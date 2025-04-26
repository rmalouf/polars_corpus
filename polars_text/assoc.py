from __future__ import annotations

from typing import overload

import numpy as np
from numba import guvectorize, int64, float64
import polars as pl

from ._typing import TPolarsFrame

__all__ = ["crosstab", "compute_mi", "compute_min_sens", "compute_loglik", "assoc"]


def crosstab(df: TPolarsFrame, x: str, y: str) -> TPolarsFrame:
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
    # Input validation
    if x not in df.columns or y not in df.columns:
        raise ValueError(f"Columns {x} and/or {y} not found in dataframe")

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


def _validated_crosstab(df: TPolarsFrame) -> TPolarsFrame:
    required_cols = ["f12", "n", "f1", "f2"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Expected: {required_cols}")
    return df.filter(
        pl.col("f1") > 0, pl.col("f2") > 0, pl.col("f12") >= 0, pl.col("n") > 0
    )


def compute_mi(table: TPolarsFrame) -> TPolarsFrame:
    """
    Computes the Pointwise Mutual Information (PMI) for the given table using its columns.

    :param table: Input table containing the necessary columns to compute PMI. The table should
        include at least four columns: `f12` (joint frequency of two events), `n` (total
        number of observations), `f1` (frequency of the first event), and `f2` (frequency of
        the second event). The columns are expected to use these exact names.
    :type table: TPolarsFrame
    :return: The same input table with an additional column `pmi` that contains the computed
        PMI values for each row.
    :rtype: TPolarsFrame
    """
    return _validated_crosstab(table).with_columns(
        pmi=((pl.col("f12") * pl.col("n")) / (pl.col("f1") * pl.col("f2"))).log()
    )


def compute_loglik(table: TPolarsFrame) -> TPolarsFrame:
    table = _validated_crosstab(table)
    data = table.with_columns(
        pl.struct(["f12", "f1", "f2", "n"])
        .map_batches(
            lambda r: _loglik(
                r.struct.field("f12"),
                r.struct.field("f1"),
                r.struct.field("f2"),
                r.struct.field("n"),
            ),
            is_elementwise=True,
        )
        .alias("loglik")
    )
    return data


@guvectorize(
    [(int64[:], int64[:], int64[:], int64[:], float64[:])],
    "(n),(n),(n),(n)->(n)",
    nopython=True,
)
def _loglik(f12, f1, f2, n, result):
    for i in range(len(f12)):
        o11 = f12[i]
        o12 = f1[i] - f12[i]
        o21 = f2[i] - f12[i]
        o22 = n[i] - f1[i] - f2[i] + f12[i]
        # r1 = f1[i]
        # r2 = n[i] - r1
        # c1 = f2[i]
        # c2 = n[i] - c1
        # e11 = r1 * c1 / n[i]
        # e12 = r1 * c2 / n[i]
        # e21 = r2 * c1 / n[i]
        # e22 = r2 * c2 / n[i]
        e11 = f1[i] * f2[i] / n[i]
        e12 = f1[i] * (n[i] - f2[i]) / n[i]
        e21 = (n[i] - f1[i]) * f2[i] / n[i]
        e22 = (n[i] - f1[i]) * (n[i] - f2[i]) / n[i]
        result[i] = 2 * (
            o11 * np.log(o11 / e11)
            + o12 * np.log(o12 / e12)
            + o21 * np.log(o21 / e21)
            + o22 * np.log(o22 / e22)
        )


def compute_min_sens(table: TPolarsFrame) -> TPolarsFrame:
    """
    Compute the mean square (ms) values for a given data table. The function
    calculates the minimum horizontal value between two derived columns,
    "f12 / f1" and "f12 / f2", and appends it as a new column named `ms` to
    the original data table.

    :param table: The input table represented as a TPolarsFrame. The table is
        expected to have columns named "f12", "f1", and "f2", with numeric
        values involved in the calculations.
    :return: A TPolarsFrame instance with an additional column `ms` containing
        the computed mean square values.
    """
    return table.with_columns(
        min_sens=pl.min_horizontal(
            pl.col("f12") / pl.col("f1"), pl.col("f12") / pl.col("f2")
        )
    )


def assoc(
    df: TPolarsFrame, x: str, y: str, method: str, min_freq: int = 0
) -> TPolarsFrame:
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
    table: TPolarsFrame = crosstab(df, x, y).filter(pl.col("f12") >= min_freq)
    match method:
        case "mi":
            return compute_mi(table)
        case "min_sens":
            return compute_min_sens(table)
        case "loglik":
            return compute_loglik(table)
        case _:
            raise ValueError(f"Unknown method: {method}")
