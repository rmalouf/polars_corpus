from __future__ import annotations

from pathlib import Path

import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from ._typing import T_Frame

__all__ = [
    "crosstab",
    "pmi",
    "minsens",
    "smp",
    "chisq",
    "loglik",
    "welchs_t",
    "welchs_t_from_stats",
]
LIB = Path(__file__).parent


def crosstab(
    df: T_Frame, x: IntoExprColumn, y: IntoExprColumn, freqs_name: str = "freqs"
) -> T_Frame:
    """
    Create a crosstabulation (contingency table) from two categorical variables.

    Computes a contingency table showing the joint frequency distribution
    of two categorical variables, along with marginal totals and grand total.
    Null values in either variable are automatically excluded from the analysis.

    Parameters
    ----------
    df : T_Frame
        Input data as a Polars DataFrame or LazyFrame containing the variables.
    x : IntoExprColumn
        Column name of the first categorical variable (row variable).
    y : IntoExprColumn
        Column name of the second categorical variable (column variable).
    freqs_name : str, default "freqs"
        Name for the output frequencies struct column.

    Returns
    -------
    T_Frame
        A DataFrame containing the contingency table with the following columns:

        - x : Levels/categories of the first variable
        - y : Levels/categories of the second variable
        - freqs : Struct with fields {f12, f1, f2, n} where:
            - f12: joint frequency (count of x,y pairs)
            - f1: row marginal (total count for this x value)
            - f2: column marginal (total count for this y value)
            - n: grand total

    Raises
    ------
    ColumnNotFoundError
        If either column `x` or `y` does not exist in the DataFrame.
    """
    f12 = pl.len().cast(pl.UInt64)
    if isinstance(x, str):
        x = pl.col(x)
    if isinstance(y, str):
        y = pl.col(y)
    x_name = x.meta.output_name()
    y_name = y.meta.output_name()
    return (
        df.select(x, y)
        .drop_nulls([x_name, y_name])
        .group_by(x_name, y_name)
        .agg(f12.alias("f12"))
        .with_columns(
            pl.struct(
                pl.col("f12"),
                pl.col("f12").sum().over(x_name).alias("f1"),
                pl.col("f12").sum().over(y_name).alias("f2"),
                pl.col("f12").sum().alias("n"),
            ).alias(freqs_name)
        )
        .drop("f12")
    )


def _validated_crosstab(df: T_Frame) -> T_Frame:
    # required_cols = ["f12", "n", "f1", "f2"]
    # if not all(col in df.columns for col in required_cols):
    #    raise ValueError(f"Missing required columns. Expected: {required_cols}")
    return df.filter(
        pl.col("f1") > 0, pl.col("f2") > 0, pl.col("f12") >= 0, pl.col("n") > 0
    )


def pmi(
    f12: IntoExprColumn, f1: IntoExprColumn, f2: IntoExprColumn, n: IntoExprColumn
) -> pl.Expr | pl.Series:
    """
    Compute Pointwise Mutual Information (PMI) for contingency table data.

    Calculates PMI values measuring the association strength between two
    categorical variables based on their observed vs. expected co-occurrence
    frequencies. PMI quantifies how much more (or less) frequently two events
    co-occur compared to what would be expected under statistical independence.

    Parameters
    ----------
    f12 : IntoExprColumn
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExprColumn
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExprColumn
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExprColumn
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the PMI values for each variable pair.

    Notes
    -----
    PMI is calculated as:

    .. math::
        PMI(x,y) = \\log\\frac{P(x,y)}{P(x)P(y)} = \\log\\frac{f_{12} \\cdot n}{f_1 \\cdot f_2}
    """

    f12 = pl.col(f12) if isinstance(f12, str) else f12
    f1 = pl.col(f1) if isinstance(f1, str) else f1
    f2 = pl.col(f2) if isinstance(f2, str) else f2
    n = pl.col(n) if isinstance(n, str) else n

    return ((f12 * n) / (f1 * f2)).log()


def chisq(
    f12: IntoExprColumn,
    f1: IntoExprColumn,
    f2: IntoExprColumn,
    n: IntoExprColumn,
    yates: bool = False,
) -> pl.Expr:
    """
    Compute Pearson's chi-squared (χ²) statistic for contingency table data.

    Calculates the chi-squared statistic measuring the association strength
    between two categorical variables by comparing observed frequencies to
    those expected under statistical independence, using the closed-form
    expression for a 2×2 table.

    Parameters
    ----------
    f12 : IntoExprColumn
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExprColumn
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExprColumn
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExprColumn
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.
    yates : bool, default False
        Whether to apply Yates' continuity correction. When True, matches the
        default behaviour of :func:`scipy.stats.chi2_contingency` for 2×2 tables.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the chi-squared values for each
        variable pair.

    Notes
    -----
    For a 2×2 contingency table the statistic reduces to:

    .. math::
        \\chi^2 = \\frac{n\\,(|O_{11} O_{22} - O_{12} O_{21}| - c)^2}
                       {f_1\\,f_2\\,(n - f_1)\\,(n - f_2)}

    where the observed cells are derived from the margins as
    :math:`O_{11} = f_{12}`, :math:`O_{12} = f_1 - f_{12}`,
    :math:`O_{21} = f_2 - f_{12}`, :math:`O_{22} = n - f_1 - f_2 + f_{12}`,
    and the continuity-correction term is :math:`c = n / 2` when `yates` is
    True and :math:`c = 0` otherwise.
    """

    # Cast to Float64 first: the cross-product difference below can go negative,
    # which would underflow for the unsigned integer counts crosstab produces.
    f12 = (pl.col(f12) if isinstance(f12, str) else f12).cast(pl.Float64)
    f1 = (pl.col(f1) if isinstance(f1, str) else f1).cast(pl.Float64)
    f2 = (pl.col(f2) if isinstance(f2, str) else f2).cast(pl.Float64)
    n = (pl.col(n) if isinstance(n, str) else n).cast(pl.Float64)

    o11 = f12
    o12 = f1 - f12
    o21 = f2 - f12
    o22 = n - f1 - f2 + f12

    det = o11 * o22 - o12 * o21
    if yates:
        det = det.abs() - n / 2
    return n * det.pow(2) / (f1 * f2 * (n - f1) * (n - f2))


def loglik(
    f12: IntoExprColumn, f1: IntoExprColumn, f2: IntoExprColumn, n: IntoExprColumn
) -> pl.Expr:
    """
    Compute log-likelihood ratio (G²) statistic for contingency table data.

    Calculates the log-likelihood ratio statistic (also known as G² or
    G-squared) which measures the association strength between two categorical
    variables by comparing observed frequencies to those expected under
    statistical independence.

    Parameters
    ----------
    f12 : IntoExprColumn
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExprColumn
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExprColumn
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExprColumn
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the log-likelihood ratio values
        for each variable pair.

    Notes
    -----
    The log-likelihood ratio statistic (G²) is calculated as:

    .. math::
        G^2 = 2 \\sum_{i,j} f_{ij} \\log\\frac{f_{ij}}{e_{ij}}

    where f_{ij} are the observed frequencies and e_{ij} are the expected
    frequencies under independence: e_{ij} = (f_{i·} × f_{·j}) / n
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=[f12, f1, f2, n],
        function_name="py_loglik",
        is_elementwise=True,
    )


def minsens(
    f12: IntoExprColumn, f1: IntoExprColumn, f2: IntoExprColumn, n: IntoExprColumn
) -> pl.Expr:
    """
    Compute minimum sensitivity values for contingency table data.

    Calculates the minimum sensitivity (minimum of precision and recall)
    as an association measure between two categorical variables. This metric
    represents the smaller of the two conditional probabilities: P(y|x) and P(x|y).

    Parameters
    ----------
    f12 : IntoExprColumn
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExprColumn
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExprColumn
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExprColumn
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression. Note: This parameter is not used in the calculation
        but is kept for consistency with other association measures.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the minimum sensitivity values
        for each variable pair.

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
    """

    f12 = pl.col(f12) if isinstance(f12, str) else f12
    f1 = pl.col(f1) if isinstance(f1, str) else f1
    f2 = pl.col(f2) if isinstance(f2, str) else f2
    n = pl.col(n) if isinstance(n, str) else n

    return pl.min_horizontal(f12 / f1, f12 / f2)


def smp(
    f12: IntoExprColumn,
    f1: IntoExprColumn,
    f2: IntoExprColumn,
    n: IntoExprColumn,
    k: float,
) -> pl.Expr:
    """
    Compute Kilgarriff's "simple maths" parameter for contingency table data.

    Calculates the ratio of a word's frequency in the target corpus to its
    frequency in the reference corpus, with a smoothing constant `k` added to
    both frequencies to avoid division by zero and to damp the effect of rare
    words.

    Parameters
    ----------
    f12 : IntoExprColumn
        Joint frequencies of variable pairs (target frequency). Can be a
        column name (str) or Polars expression.
    f1 : IntoExprColumn
        Marginal frequencies of first variable (target + reference frequency).
        Can be a column name (str) or Polars expression.
    f2 : IntoExprColumn
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression. Note: This parameter is not used in the calculation
        but is kept for consistency with other association measures.
    n : IntoExprColumn
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression. Note: This parameter is not used in the calculation
        but is kept for consistency with other association measures.
    k : float
        Smoothing constant added to both the target and reference frequencies.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the simple maths values for each
        variable pair.

    Notes
    -----
    Simple maths is calculated as:

    .. math::
        \\text{smp}(x,y) = \\frac{f_{12} + k}{(f_1 - f_{12}) + k}

    where :math:`f_1 - f_{12}` is the frequency of the word in the reference
    corpus.

    References
    ----------
    Kilgarriff, A. (2009, July). Simple maths for keywords. In Proceedings of
    the Corpus Linguistics Conference. Liverpool, UK.
    """

    f12 = pl.col(f12) if isinstance(f12, str) else f12
    f1 = pl.col(f1) if isinstance(f1, str) else f1
    f2 = pl.col(f2) if isinstance(f2, str) else f2
    n = pl.col(n) if isinstance(n, str) else n

    return (f12 + k) / (f1 - f12 + k)


def welchs_t(x1: IntoExprColumn, x2: IntoExprColumn, alt: str = "twosided") -> pl.Expr:
    """
    Perform Welch's t-test for equality of two independent samples with unequal variances.

    Welch's t-test is a variation of Student's t-test that does not assume equal
    population variances. It compares the means of two independent samples to
    determine if they are statistically different from each other.

    Parameters
    ----------
    x1 : IntoExprColumn
        First sample data. Can be a column name (str) or Polars expression.
    x2 : IntoExprColumn
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
    """
    s1 = pl.col(x1) if isinstance(x1, str) else x1
    s2 = pl.col(x2) if isinstance(x2, str) else x2

    if alt not in ["twosided", "greater", "less"]:
        raise ValueError(f"Unknown alternative value: {alt}")

    return register_plugin_function(
        plugin_path=LIB,
        args=[s1, s2],
        function_name="py_welchs_t",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"alt": alt},
    ).alias("t_test")


def welchs_t_from_stats(
    s1: IntoExprColumn,
    ss1: IntoExprColumn,
    n1: IntoExprColumn,
    s2: IntoExprColumn,
    ss2: IntoExprColumn,
    n2: IntoExprColumn,
    alt: str = "twosided",
) -> pl.Expr:
    """
    Perform Welch's t-test using pre-computed summary statistics.

    This function performs Welch's t-test for equality of two independent samples
    using pre-computed summary statistics (means, sum of squares, and sample sizes)
    rather than raw data.

    Parameters
    ----------
    s1 : IntoExprColumn
        Sum (or mean × n) of the first sample. Can be a column name (str) or
        Polars expression.
    ss1 : IntoExprColumn
        Sum of squares of the first sample. Can be a column name (str) or
        Polars expression.
    n1 : IntoExprColumn
        Sample size of the first sample. Can be a column name (str) or
        Polars expression.
    s2 : IntoExprColumn
        Sum (or mean × n) of the second sample. Can be a column name (str) or
        Polars expression.
    ss2 : IntoExprColumn
        Sum of squares of the second sample. Can be a column name (str) or
        Polars expression.
    n2 : IntoExprColumn
        Sample size of the second sample. Can be a column name (str) or
        Polars expression.
    alt : {'twosided', 'greater', 'less'}, default 'twosided'
        Alternative hypothesis to test:

        - 'twosided' : the means are unequal (two-tailed test)
        - 'greater' : the mean of the first sample is greater than the second (one-tailed)
        - 'less' : the mean of the first sample is less than the second (one-tailed)

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
    The sample means and variances are computed from the summary statistics as:

    .. math::
        \\bar{x}_i = \\frac{s_i}{n_i}

    .. math::
        \\text{var}_i = \\frac{ss_i - \\frac{s_i^2}{n_i}}{n_i - 1}

    The test statistic and degrees of freedom are then calculated using the
    same formulas as in `welchs_t`.
    """
    s1_s = pl.col(s1) if isinstance(s1, str) else s1
    ss1_s = pl.col(ss1) if isinstance(ss1, str) else ss1
    n1_s = pl.col(n1) if isinstance(n1, str) else n1
    s2_s = pl.col(s2) if isinstance(s2, str) else s2
    ss2_s = pl.col(ss2) if isinstance(ss2, str) else ss2
    n2_s = pl.col(n2) if isinstance(n2, str) else n2

    if alt not in ["twosided", "greater", "less"]:
        raise ValueError(f"Unknown alternative value: {alt}")

    return register_plugin_function(
        plugin_path=LIB,
        args=[s1_s, ss1_s, n1_s, s2_s, ss2_s, n2_s],
        function_name="py_welchs_t_from_stats",
        is_elementwise=True,
        returns_scalar=False,
        kwargs={"alt": alt},
    ).alias("t_test")
