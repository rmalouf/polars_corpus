from __future__ import annotations

from typing import overload

import polars as pl

from ._typing import TPolarsFrame

all = ["crosstab", "compute_pmi", "compute_ms", "assoc"]


def crosstab(df: TPolarsFrame, x: str, y: str) -> TPolarsFrame:
    """
    Creates a crosstabulation (contingency table) from a given dataframe with two variables.
    The crosstab includes frequencies of occurrence as well as marginal and total sums.

    :param df: The input data as a Polars DataFrame or LazyFrame.
    :param x: The column name of the first variable (independent variable).
    :param y: The column name of the second variable (dependent variable).
    :return: A Polars DataFrame containing the contingency table. The result includes:
             - x: Levels of the first variable.
             - y: Levels of the second variable.
             - f12: Joint frequencies (count of occurrences for each combination of x and y).
             - f1: Row marginal sums (frequency sums grouped by x).
             - f2: Column marginal sums (frequency sums grouped by y).
             - n: Grand total (the sum of all frequencies).
    """
    t = (
        df.select(x, y)
        .group_by(x, y)
        .len("f12")
        .with_columns(
            pl.col("f12").sum().over(x).alias("f1"),
            pl.col("f12").sum().over(y).alias("f2"),
            pl.col("f12").sum().alias("n"),
        )
    )
    return t.select(x, y, "f12", "f1", "f2", "n")


def compute_pmi(table: TPolarsFrame) -> TPolarsFrame:
    """
    Computes the Pointwise Mutual Information (PMI) for the given table using its columns.
    The PMI is computed as the logarithm of the ratio of the joint probability to the product
    of the marginals for two events. This is often used in statistical analysis and information
    theory to measure the association between variables.

    :param table: Input table containing the necessary columns to compute PMI. The table should
        include at least four columns: `f12` (joint frequency of two events), `n` (total
        number of observations), `f1` (frequency of the first event), and `f2` (frequency of
        the second event). The columns are expected to use these exact names.
    :type table: TPolarsFrame
    :return: The same input table with an additional column `pmi` that contains the computed
        PMI values for each row.
    :rtype: TPolarsFrame
    """
    return table.with_columns(
        pmi=((pl.col("f12") * pl.col("n")) / (pl.col("f1") * pl.col("f2"))).log()
    )


def compute_ms(table: TPolarsFrame) -> TPolarsFrame:
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
        ms=pl.min_horizontal(pl.col("f12") / pl.col("f1"), pl.col("f12") / pl.col("f2"))
    )


def assoc(
    df: TPolarsFrame, x: str, y: str, method: str, min_freq: int = 0
) -> TPolarsFrame:
    """
    Calculate association metrics between two categorical variables in a dataframe.

    This function computes specific association metrics between two categorical
    variables ('x' and 'y') from a given dataframe. It supports the calculation
    of either Pointwise Mutual Information (PMI) or Mutual Strength (MS) based
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
        case "pmi":
            return compute_pmi(table)
        case "ms":
            return compute_ms(table)
        case _:
            raise ValueError(f"Unknown method: {method}")
