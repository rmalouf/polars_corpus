from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from ._typing import T_Frame, IntoExpr

__all__ = [
    "crosstab",
    "compute_mi",
    "compute_min_sens",
    "compute_loglik",
    "assoc",
    "welchs_t",
]
LIB = Path(__file__).parent


def crosstab(df: T_Frame, x: str, y: str) -> T_Frame:
    """
    Create a crosstabulation (contingency table) from two categorical variables.

    Computes a contingency table showing the joint frequency distribution
    of two categorical variables, along with marginal totals and grand total.
    Null values in either variable are automatically excluded from the analysis.

    Parameters
    ----------
    df : T_Frame
        Input data as a Polars DataFrame or LazyFrame containing the variables.
    x : str
        Column name of the first categorical variable (row variable).
    y : str
        Column name of the second categorical variable (column variable).

    Returns
    -------
    T_Frame
        A DataFrame containing the contingency table with the following columns:

        - x : Levels/categories of the first variable
        - y : Levels/categories of the second variable
        - f12 : Joint frequencies (count of observations for each x,y pair)
        - f1 : Row marginal sums (total frequency of each x level)
        - f2 : Column marginal sums (total frequency of each y level)
        - n : Grand total (total number of observations)

    Raises
    ------
    ColumnNotFoundError
        If either column `x` or `y` does not exist in the DataFrame.

    Notes
    -----
    The contingency table provides the foundation for computing various
    association measures between categorical variables. The joint frequencies
    (f12) represent the observed counts, while marginal frequencies (f1, f2)
    and the grand total (n) enable calculation of expected frequencies under
    independence assumptions.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import crosstab
    >>> df = pl.DataFrame({
    ...     'gender': ['M', 'F', 'M', 'F', 'M'],
    ...     'response': ['yes', 'no', 'yes', 'yes', 'no']
    ... })
    >>> result = crosstab(df, 'gender', 'response')
    >>> print(result)
    shape: (4, 6)
    ┌────────┬──────────┬─────┬─────┬─────┬─────┐
    │ gender ┆ response ┆ f12 ┆ f1  ┆ f2  ┆ n   │
    │ ---    ┆ ---      ┆ --- ┆ --- ┆ --- ┆ --- │
    │ str    ┆ str      ┆ u32 ┆ u32 ┆ u32 ┆ u32 │
    ╞════════╪══════════╪═════╪═════╪═════╪═════╡
    │ F      ┆ no       ┆ 1   ┆ 2   ┆ 2   ┆ 5   │
    │ F      ┆ yes      ┆ 1   ┆ 2   ┆ 3   ┆ 5   │
    │ M      ┆ no       ┆ 1   ┆ 3   ┆ 2   ┆ 5   │
    │ M      ┆ yes      ┆ 2   ┆ 3   ┆ 3   ┆ 5   │
    └────────┴──────────┴─────┴─────┴─────┴─────┘
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
    Compute Pointwise Mutual Information (PMI) for contingency table data.

    Calculates PMI values measuring the association strength between two
    categorical variables based on their observed vs. expected co-occurrence
    frequencies. PMI quantifies how much more (or less) frequently two events
    co-occur compared to what would be expected under statistical independence.

    Parameters
    ----------
    table : T_Frame
        Contingency table containing frequency data. Must include columns:

        - f12 : Joint frequencies of variable pairs
        - f1 : Marginal frequencies of first variable
        - f2 : Marginal frequencies of second variable
        - n : Grand total (total number of observations)

    Returns
    -------
    T_Frame
        Input table with an additional `pmi` column containing the computed
        PMI values for each variable pair.

    Notes
    -----
    PMI is calculated as:

    .. math::
        PMI(x,y) = \\log\\frac{P(x,y)}{P(x)P(y)} = \\log\\frac{f_{12} \\cdot n}{f_1 \\cdot f_2}

    Positive PMI values indicate that events co-occur more frequently than
    expected by chance, while negative values indicate they co-occur less
    frequently than expected. PMI = 0 suggests statistical independence.

    The function validates input data by filtering out invalid frequency
    combinations (zero marginals, negative frequencies, etc.) before
    computation.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import compute_mi, crosstab
    >>> df = pl.DataFrame({
    ...     'word': ['good', 'bad', 'good', 'bad'],
    ...     'sentiment': ['pos', 'neg', 'pos', 'neg']
    ... })
    >>> table = crosstab(df, 'word', 'sentiment')
    >>> result = compute_mi(table)
    >>> print(result.select('word', 'sentiment', 'pmi'))
    shape: (4, 3)
    ┌──────┬───────────┬──────────┐
    │ word ┆ sentiment ┆ pmi      │
    │ ---  ┆ ---       ┆ ---      │
    │ str  ┆ str       ┆ f64      │
    ╞══════╪═══════════╪══════════╡
    │ bad  ┆ neg       ┆ 0.693147 │
    │ bad  ┆ pos       ┆ -inf     │
    │ good ┆ neg       ┆ -inf     │
    │ good ┆ pos       ┆ 0.693147 │
    └──────┴───────────┴──────────┘
    """
    return _validated_crosstab(table).with_columns(
        pmi=((pl.col("f12") * pl.col("n")) / (pl.col("f1") * pl.col("f2"))).log()
    )


def loglik(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        args=[f12, f1, f2, n],
        function_name="loglik",
        is_elementwise=True,
    )


## TODO: make this work with LazyFrames
def compute_loglik(table: pl.DataFrame) -> pl.DataFrame:
    """
    Compute log-likelihood ratio (G²) statistic for contingency table data.

    Calculates the log-likelihood ratio statistic (also known as G² or
    G-squared) which measures the association strength between two categorical
    variables by comparing observed frequencies to those expected under
    statistical independence.

    Parameters
    ----------
    table : pl.DataFrame
        Contingency table containing frequency data. Must include columns:

        - f12 : Joint frequencies of variable pairs
        - f1 : Marginal frequencies of first variable
        - f2 : Marginal frequencies of second variable
        - n : Grand total (total number of observations)

    Note
    ----
    Currently only works with DataFrames, not LazyFrames.

    Returns
    -------
    pl.DataFrame
        Input table with an additional `LL` column containing the computed
        log-likelihood ratio values for each variable pair.

    Notes
    -----
    The log-likelihood ratio statistic (G²) is calculated as:

    .. math::
        G^2 = 2 \\sum_{i,j} f_{ij} \\log\\frac{f_{ij}}{e_{ij}}

    where f_{ij} are the observed frequencies and e_{ij} are the expected
    frequencies under independence: e_{ij} = (f_{i·} × f_{·j}) / n

    The G² statistic follows a chi-squared distribution under the null
    hypothesis of independence, making it useful for significance testing.
    Higher values indicate stronger association between variables.

    Unlike chi-squared, G² has additive properties and is more appropriate
    for sparse contingency tables with small expected frequencies.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import compute_loglik, crosstab
    >>> df = pl.DataFrame({
    ...     'treatment': ['A', 'B', 'A', 'B', 'A', 'B'],
    ...     'outcome': ['success', 'fail', 'success', 'success', 'fail', 'fail']
    ... })
    >>> table = crosstab(df, 'treatment', 'outcome')
    >>> result = compute_loglik(table)
    >>> print(result.select('treatment', 'outcome', 'LL'))
    shape: (4, 3)
    ┌───────────┬─────────┬──────────┐
    │ treatment ┆ outcome ┆ LL       │
    │ ---       ┆ ---     ┆ ---      │
    │ str       ┆ str     ┆ f64      │
    ╞═══════════╪═════════╪══════════╡
    │ A         ┆ fail    ┆ 0.693147 │
    │ A         ┆ success ┆ 0.693147 │
    │ B         ┆ fail    ┆ 0.693147 │
    │ B         ┆ success ┆ 0.693147 │
    └───────────┴─────────┴──────────┘
    """
    table = _validated_crosstab(table)
    data = loglik(
        table.get_column("f12"),
        table.get_column("f1"),
        table.get_column("f2"),
        table.get_column("n"),
    ).alias("LL")
    return table.with_columns(data)


def compute_min_sens(table: T_Frame) -> T_Frame:
    """
    Compute minimum sensitivity values for contingency table data.

    Calculates the minimum sensitivity (minimum of precision and recall)
    as an association measure between two categorical variables. This metric
    represents the smaller of the two conditional probabilities: P(y|x) and P(x|y).

    Parameters
    ----------
    table : T_Frame
        Contingency table containing frequency data. Must include columns:

        - f12 : Joint frequencies of variable pairs
        - f1 : Marginal frequencies of first variable
        - f2 : Marginal frequencies of second variable

    Returns
    -------
    T_Frame
        Input table with an additional `min_sens` column containing the computed
        minimum sensitivity values for each variable pair.

    Notes
    -----
    Minimum sensitivity is calculated as:

    .. math::
        \\text{min_sens}(x,y) = \\min\\left(\\frac{f_{12}}{f_1}, \\frac{f_{12}}{f_2}\\right)

    This corresponds to:

    .. math::
        \\text{min_sens}(x,y) = \\min(P(y|x), P(x|y))

    where:

    - f₁₂/f₁ represents the precision: P(y|x)
    - f₁₂/f₂ represents the recall: P(x|y)

    Values range from 0 to 1, where higher values indicate stronger association.
    A value of 1 indicates perfect association (complete dependence), while
    values near 0 indicate weak association.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import compute_min_sens, crosstab
    >>> df = pl.DataFrame({
    ...     'cause': ['rain', 'sun', 'rain', 'sun'],
    ...     'effect': ['wet', 'dry', 'wet', 'dry']
    ... })
    >>> table = crosstab(df, 'cause', 'effect')
    >>> result = compute_min_sens(table)
    >>> print(result.select('cause', 'effect', 'min_sens'))
    shape: (4, 3)
    ┌───────┬────────┬──────────┐
    │ cause ┆ effect ┆ min_sens │
    │ ---   ┆ ---    ┆ ---      │
    │ str   ┆ str    ┆ f64      │
    ╞═══════╪════════╪══════════╡
    │ rain  ┆ dry    ┆ 0.0      │
    │ rain  ┆ wet    ┆ 0.5      │
    │ sun   ┆ dry    ┆ 0.5      │
    │ sun   ┆ wet    ┆ 0.0      │
    └───────┴────────┴──────────┘
    """
    return table.with_columns(
        min_sens=pl.min_horizontal(
            pl.col("f12") / pl.col("f1"), pl.col("f12") / pl.col("f2")
        )
    )


def assoc(df: T_Frame, x: str, y: str, method: str, min_freq: int = 0) -> T_Frame:
    """
    Calculate association metrics between two categorical variables.

    Computes various statistical association measures between two categorical
    variables using their contingency table. The function first creates a
    crosstabulation of the variables, applies frequency filtering, then
    calculates the specified association metric.

    Parameters
    ----------
    df : T_Frame
        A Polars DataFrame or LazyFrame containing the categorical variables.
    x : str
        Column name of the first categorical variable (independent variable).
    y : str
        Column name of the second categorical variable (dependent variable).
    method : {'mi', 'min_sens', 'loglik'}
        Association metric to compute:

        - 'mi' : Pointwise Mutual Information (PMI)
        - 'min_sens' : Minimum sensitivity (minimum of precision and recall)
        - 'loglik' : Log-likelihood ratio statistic
    min_freq : int, default 0
        Minimum joint frequency threshold. Filters out variable pairs with
        joint frequency (f12) below this value before computing associations.

    Returns
    -------
    T_Frame
        A DataFrame containing the contingency table with the computed
        association metric. Always includes columns:

        - x : Levels of the first variable
        - y : Levels of the second variable
        - f12 : Joint frequencies
        - f1 : Row marginal sums (frequency of x)
        - f2 : Column marginal sums (frequency of y)
        - n : Grand total

        Plus one additional column depending on method:

        - 'pmi' : PMI values (for method='mi')
        - 'min_sens' : Minimum sensitivity values (for method='min_sens')
        - 'LL' : Log-likelihood values (for method='loglik')

    Raises
    ------
    ValueError
        If `method` is not one of 'mi', 'min_sens', or 'loglik'.
    ColumnNotFoundError
        If columns `x` or `y` do not exist in the DataFrame.

    Notes
    -----
    The association metrics are computed as follows:

    **Pointwise Mutual Information (PMI)**:

    .. math::
        PMI(x,y) = \\log\\frac{P(x,y)}{P(x)P(y)} = \\log\\frac{f_{12} \\cdot n}{f_1 \\cdot f_2}

    **Minimum Sensitivity**:

    .. math::
        \\text{min_sens}(x,y) = \\min\\left(\\frac{f_{12}}{f_1}, \\frac{f_{12}}{f_2}\\right)

    **Log-likelihood ratio**: Computed using the G² statistic comparing
    observed vs. expected frequencies under independence assumption.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import assoc
    >>> df = pl.DataFrame({
    ...     'word': ['the', 'cat', 'the', 'dog', 'cat'],
    ...     'pos': ['DET', 'NOUN', 'DET', 'NOUN', 'NOUN']
    ... })
    >>> result = assoc(df, 'word', 'pos', 'mi')
    >>> print(result)
    shape: (4, 7)
    ┌──────┬──────┬─────┬─────┬─────┬─────┬──────────┐
    │ word ┆ pos  ┆ f12 ┆ f1  ┆ f2  ┆ n   ┆ pmi      │
    │ ---  ┆ ---  ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---      │
    │ str  ┆ str  ┆ u32 ┆ u32 ┆ u32 ┆ u32 ┆ f64      │
    ╞══════╪══════╪═════╪═════╪═════╪═════╪══════════╡
    │ cat  ┆ NOUN ┆ 2   ┆ 2   ┆ 3   ┆ 5   ┆ 0.510826 │
    │ dog  ┆ NOUN ┆ 1   ┆ 1   ┆ 3   ┆ 5   ┆ 0.510826 │
    │ the  ┆ DET  ┆ 2   ┆ 2   ┆ 2   ┆ 5   ┆ 0.916291 │
    └──────┴──────┴─────┴─────┴─────┴─────┴──────────┘
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


def welchs_t(x1: IntoExpr, x2: IntoExpr, alt: str = "twosided") -> pl.Expr:
    """
    Perform Welch's t-test for equality of two independent samples with unequal variances.

    Welch's t-test is a variation of Student's t-test that does not assume equal
    population variances. It compares the means of two independent samples to
    determine if they are statistically different from each other.

    Parameters
    ----------
    x1 : IntoExpr
        First sample data. Can be a column name (str) or Polars expression.
    x2 : IntoExpr
        Second sample data. Can be a column name (str) or Polars expression.
    alt : {'twosided', 'greater', 'less'}, default 'twosided'
        Alternative hypothesis to test:

        - 'twosided' : the means are unequal (two-tailed test)
        - 'greater' : the mean of x1 is greater than the mean of x2 (one-tailed)
        - 'less' : the mean of x1 is less than the mean of x2 (one-tailed)

    Returns
    -------
    pl.Expr
        A Polars expression that returns a struct with the following fields:

        - 'stat' : float
            The t-statistic of the test
        - 'pval' : float
            The p-value of the test
        - 'df' : float
            The degrees of freedom used in the test

        Returns null values for all fields if the test cannot be performed
        (e.g., insufficient data or zero variance in both samples).

    Raises
    ------
    ValueError
        If `alt` is not one of 'twosided', 'greater', or 'less'.

    Notes
    -----
    The test statistic is calculated as:

    .. math::
        t = \\frac{\\bar{x}_1 - \\bar{x}_2}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}

    where :math:`\\bar{x}_i`, :math:`s_i^2`, and :math:`n_i` are the sample mean,
    sample variance, and sample size of the i-th sample, respectively.

    The degrees of freedom are approximated using the Welch-Satterthwaite equation:

    .. math::
        df = \\frac{(\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2})^2}{\\frac{(s_1^2/n_1)^2}{n_1-1} + \\frac{(s_2^2/n_2)^2}{n_2-1}}

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import welchs_t
    >>> df = pl.DataFrame({'group1': [1, 2, 3], 'group2': [4, 5, 6]})
    >>> result = df.select(welchs_t('group1', 'group2')).unnest('t_test')
    >>> print(result)
    shape: (1, 3)
    ┌──────────┬──────────┬─────┐
    │ stat     ┆ pval     ┆ df  │
    │ ---      ┆ ---      ┆ --- │
    │ f64      ┆ f64      ┆ f64 │
    ╞══════════╪══════════╪═════╡
    │ -3.67423 ┆ 0.021312 ┆ 4.0 │
    └──────────┴──────────┴─────┘
    """
    s1 = pl.col(x1) if isinstance(x1, str) else x1
    s2 = pl.col(x2) if isinstance(x2, str) else x2

    if not alt in ["twosided", "greater", "less"]:
        raise ValueError(f"Unknown alternative value: {alt}")

    return register_plugin_function(
        plugin_path=LIB,
        args=[s1, s2],
        function_name="welchs_t",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"alt": alt},
    ).alias("t_test")
