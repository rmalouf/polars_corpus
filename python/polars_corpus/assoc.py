from __future__ import annotations

from pathlib import Path

import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_expr,
    collect_like,
    drop_null_rows,
)

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

ALTERNATIVES = ("twosided", "greater", "less")


def _variable_name(lf: pl.LazyFrame, var: IntoExprColumn, param: str) -> str:
    """Name the column `var` gives, checking the corpus can evaluate it.

    Unlike most of the package `crosstab` takes a Series: both variables are
    evaluated against the one corpus, so a Series of matching height lines up
    with it, and carries the name of the column it stands in for.
    """
    if isinstance(var, pl.Series):
        return var.name
    return check_expr(lf, as_expr(var, param), param=param)


def _as_freqs(
    f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr
) -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    """Turn the four contingency table arguments into Float64 expressions.

    Float64 because the measures take differences of the margins, which would
    underflow for the unsigned integer counts `crosstab` produces.
    """
    return (
        as_expr(f12, "f12").cast(pl.Float64),
        as_expr(f1, "f1").cast(pl.Float64),
        as_expr(f2, "f2").cast(pl.Float64),
        as_expr(n, "n").cast(pl.Float64),
    )


def crosstab(
    corpus: T_Frame,
    x: IntoExprColumn,
    y: IntoExprColumn,
    freqs_name: str = "freqs",
) -> T_Frame:
    """
    Create a crosstabulation (contingency table) from two categorical variables.

    Computes a contingency table showing the joint frequency distribution
    of two categorical variables, along with marginal totals and grand total.
    Null values in either variable are automatically excluded from the analysis.

    Parameters
    ----------
    corpus : T_Frame
        Input data as a Polars DataFrame or LazyFrame containing the variables.
    x : IntoExprColumn
        Column name, expression or Series giving the first categorical
        variable (row variable).
    y : IntoExprColumn
        Column name, expression or Series giving the second categorical
        variable (column variable).
    freqs_name : str, default "freqs"
        Name for the output frequencies struct column.

    Returns
    -------
    T_Frame
        The contingency table, eager if `corpus` is a DataFrame and lazy if it
        is a LazyFrame, with the following columns:

        - x : Levels/categories of the first variable
        - y : Levels/categories of the second variable
        - freqs : Struct with fields {f12, f1, f2, n} where:
            - f12: joint frequency (count of x,y pairs)
            - f1: row marginal (total count for this x value)
            - f2: column marginal (total count for this y value)
            - n: grand total

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, or is empty; or if
        `x` or `y` does not name a single column of it.

    Warns
    -----
    UserWarning
        If rows are dropped for holding a null in `x` or `y`. Raised only for
        an eager corpus: counting the dropped rows of a LazyFrame would mean
        reading it before the caller has asked for anything.
    """
    lf = as_corpus(corpus)
    x_name = _variable_name(lf, x, "x")
    y_name = _variable_name(lf, y, "y")

    # Read only the two variables, so a null elsewhere in the corpus is none of
    # this table's business, and a column of its own named `f12` cannot collide.
    counts = drop_null_rows(lf.select(x, y), corpus)
    result = (
        counts.group_by(x_name, y_name)
        .agg(pl.len().cast(pl.UInt64).alias("f12"))
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
    return collect_like(result, corpus)


def pmi(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute Pointwise Mutual Information (PMI) for contingency table data.

    Calculates PMI values measuring the association strength between two
    categorical variables based on their observed vs. expected co-occurrence
    frequencies. PMI quantifies how much more (or less) frequently two events
    co-occur compared to what would be expected under statistical independence.

    Parameters
    ----------
    f12 : IntoExpr
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExpr
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExpr
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExpr
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the PMI values for each variable pair.

    Notes
    -----
    PMI is calculated as:

    $$
    \\text{PMI}(x,y) = \\log\\frac{P(x,y)}{P(x)\\,P(y)} = \\log\\frac{f_{12} \\cdot n}{f_1 \\cdot f_2}
    $$

    References
    ----------
    - Church, K. W. and P. Hanks. 1990. Word association norms, mutual information, and
      lexicography. *Computational Linguistics* 16(1): 22-29.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    return ((f12 * n) / (f1 * f2)).log()


def chisq(
    f12: IntoExpr,
    f1: IntoExpr,
    f2: IntoExpr,
    n: IntoExpr,
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
    f12 : IntoExpr
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExpr
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExpr
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExpr
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

    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)

    o11 = f12
    o12 = f1 - f12
    o21 = f2 - f12
    o22 = n - f1 - f2 + f12

    det = o11 * o22 - o12 * o21
    if yates:
        det = det.abs() - n / 2
    return n * det.pow(2) / (f1 * f2 * (n - f1) * (n - f2))


def loglik(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute log-likelihood ratio (G²) statistic for contingency table data.

    Calculates the log-likelihood ratio statistic (also known as G² or
    G-squared) which measures the association strength between two categorical
    variables by comparing observed frequencies to those expected under
    statistical independence.

    Parameters
    ----------
    f12 : IntoExpr
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExpr
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExpr
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExpr
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
        args=list(_as_freqs(f12, f1, f2, n)),
        function_name="py_loglik",
        is_elementwise=True,
    )


def minsens(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute minimum sensitivity values for contingency table data.

    Calculates the minimum sensitivity (minimum of precision and recall)
    as an association measure between two categorical variables. This metric
    represents the smaller of the two conditional probabilities: P(y|x) and P(x|y).

    Parameters
    ----------
    f12 : IntoExpr
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExpr
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExpr
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExpr
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the minimum sensitivity values
        for each variable pair.

    Notes
    -----
    Minimum sensitivity is calculated as:

    $$
    \\begin{align*}
    \\text{minsens}(x,y) &= \\min(P(y|x), P(x|y))\\\\
    &= \\min\\left(\\frac{f_{12}}{f_1}, \\frac{f_{12}}{f_2}\\right)
    \\end{align*}
    $$

    References
    ----------
    - Wiechmann, D. 2008. On the computation of collostruction strength: Testing measures of
      association as expressions of lexical bias. *Corpus Linguistics and Linguistic
      Theory* 4(2): 253–290.
    """
    f12, f1, f2, _ = _as_freqs(f12, f1, f2, n)
    precision, recall = f12 / f1, f12 / f2
    # min_horizontal skips nulls, which would leave the other ratio standing in
    # for the minimum; the measure propagates them like every other one here.
    return (
        pl.when(precision.is_null() | recall.is_null())
        .then(None)
        .otherwise(pl.min_horizontal(precision, recall))
    )


def smp(
    f12: IntoExpr,
    f1: IntoExpr,
    f2: IntoExpr,
    n: IntoExpr,
    k: float,
) -> pl.Expr:
    """
    Compute Kilgarriff's "simple maths" parameter for contingency table data.

    Calculates the ratio of a word's frequency in the target corpus to its
    frequency in the reference corpus, with a smoothing constant `k` added to
    both frequencies to avoid division by zero and to reduce the effect of rare
    words.

    Parameters
    ----------
    f12 : IntoExpr
        Joint frequencies of variable pairs. Can be a column name (str) or
        Polars expression.
    f1 : IntoExpr
        Marginal frequencies of first variable. Can be a column name (str) or
        Polars expression.
    f2 : IntoExpr
        Marginal frequencies of second variable. Can be a column name (str) or
        Polars expression.
    n : IntoExpr
        Grand total (total number of observations). Can be a column name (str) or
        Polars expression.
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

    $$
    \\text{smp}(x,y) = \\frac{f_{12} + k}{(f_1 - f_{12}) + k}
    $$

    where $f_1-f_{12}$ is the frequency of the word in the reference
    corpus.

    References
    ----------
    - Kilgarriff, A. 2009. Simple maths for keywords. In *Proceedings of
      the Corpus Linguistics Conference.* Liverpool, UK.
    """
    f12, f1, _, _ = _as_freqs(f12, f1, f2, n)
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
        Nulls are left out of the sample rather than counted in its size.
    x2 : IntoExprColumn
        Second sample data. Can be a column name (str) or Polars expression.
        Nulls are left out of the sample rather than counted in its size.
    alt : {'twosided', 'greater', 'less'}, default 'twosided'
        Alternative hypothesis to test:

        - 'twosided' : the means are unequal (two-tailed test)
        - 'greater' : the mean of x1 is greater than the mean of x2 (one-tailed)
        - 'less' : the mean of x1 is less than the mean of x2 (one-tailed)

    Returns
    -------
    pl.Expr
        A Polars expression that returns a struct with the following fields:

        - 't' : float
            The t-statistic of the test
        - 'p' : float
            The p-value of the test
        - 'df' : float
            The degrees of freedom used in the test
        - 'g' : float
            Hedges' g, a measure of effect size

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

    The effect size is Hedges' g: Cohen's d recovered from the t-statistic, then
    scaled by the small-sample bias correction :math:`J`:

    .. math::
        d = t \\sqrt{\\frac{2\\left(\\frac{s_1^2}{n_1}+\\frac{s_2^2}{n_2}\\right)}{s_1^2+s_2^2}}
        \\qquad
        g = J d, \\quad J = 1 - \\frac{3}{4\\,df - 1}
    """
    alt = check_choice(alt, ALTERNATIVES, param="alt")
    return register_plugin_function(
        plugin_path=LIB,
        args=[x1, x2],
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
    using pre-computed means, sums of squares, and sample sizes rather than raw data.

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

        - 't' : float
            The t-statistic of the test
        - 'p' : float
            The p-value of the test
        - 'df' : float
            The degrees of freedom used in the test
        - 'g' : float
            Hedges' g, the difference in means in standard deviations

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

    The test statistic, degrees of freedom, and effect size are then calculated
    using the same formulas as in `welchs_t`.
    """
    alt = check_choice(alt, ALTERNATIVES, param="alt")
    return register_plugin_function(
        plugin_path=LIB,
        args=[s1, ss1, n1, s2, ss2, n2],
        function_name="py_welchs_t_from_stats",
        is_elementwise=True,
        returns_scalar=False,
        kwargs={"alt": alt},
    ).alias("t_test")
