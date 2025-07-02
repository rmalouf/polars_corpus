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
    "get_contexts",
    "compute_mi",
    "compute_min_sens",
    "compute_loglik",
    "assoc",
]
LIB = Path(__file__).parent


def get_contexts(
    df: T_Frame,
    by: IntoExprColumn,
    width: Optional[int] = None,
    left_width: Optional[int] = None,
    right_width: Optional[int] = None,
) -> T_Frame:
    """
    Generate a DataFrame containing the context around a column of values by
    shifting values left and right. The context size can be specified
    symmetrically or asymmetrically based on given width parameters.

    :param df:
        The input DataFrame to process. Must be a TPolarsFrame.
    :param by:
        The column (or expression to select a column) used to generate
        context values. Accepts a string column name or an instance of
        IntoExprColumn.
    :param width:
        The symmetrical width of the context around the target column. If
        specified, both `left_width` and `right_width` are inferred based
        on this value (i.e., left_width = right_width = width). Cannot be
        specified along with `left_width` or `right_width` individually.
    :param left_width:
        The width of the context to the left of the target column. Must be
        non-negative. Should not be used together with `width`.
    :param right_width:
        The width of the context to the right of the target column. Must be
        non-negative. Should not be used together with `width`.
    :return:
        A transformed DataFrame containing the context values for the target
        column, including shifted columns prefixed with 'context-' (left)
        and 'context+' (right), as well as the original column labeled
        'node'.
    :rtype:
        TPolarsFrame
    """
    if width:
        if left_width is not None or right_width is not None:
            raise ValueError(
                "left_width and right_width cannot be specified when width is specified"
            )
        else:
            left_width = width
            right_width = width

    if left_width is None or right_width is None:
        raise ValueError("left_width and right_width must be specified")

    if left_width < 0 or right_width < 0:
        raise ValueError("left_width and right_width must be non-negative")

    if isinstance(by, str):
        by = pl.col(by)

    w = df.select(
        [by.shift(i).alias(f"context-{i}") for i in range(left_width, 0, -1)]
        + [by.alias("node")]
        + [by.shift(-i).alias(f"context+{i}") for i in range(1, right_width + 1)]
    )

    return w


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
