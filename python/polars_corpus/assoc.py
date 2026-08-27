from __future__ import annotations

from pathlib import Path

import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from ._typing import IntoExpr, Measure, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_expr,
    collect_like,
)

__all__ = [
    "crosstab",
    "pmi",
    "mi3",
    "logdice",
    "tscore",
    "zscore",
    "minsens",
    "smp",
    "chisq",
    "loglik",
    "welchs_t",
    "welchs_t_from_stats",
]
LIB = Path(__file__).parent

ALTERNATIVES = ("twosided", "greater", "less")

# The four contingency table counts every measure is written over, in the
# order they are passed. Also the column names `crosstab` gives them.
FREQS = ("f12", "f1", "f2", "n")

# Names polars gives an expression that was never aliased.
_DEFAULT_NAMES = FREQS + ("literal",)


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


def crosstab(corpus: T_Frame, x: IntoExprColumn, y: IntoExprColumn) -> T_Frame:
    """
    Create a cross-tabulation (contingency table) from two categorical variables.

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

    Returns
    -------
    T_Frame
        The contingency table with the following columns:

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
        `x` or `y` does not name a column.
    """
    lf = as_corpus(corpus)

    x_name = (
        x.name
        if isinstance(x, pl.Series)
        else check_expr(lf, as_expr(x, "x"), param="x")
    )
    y_name = (
        y.name
        if isinstance(y, pl.Series)
        else check_expr(lf, as_expr(y, "y"), param="y")
    )

    counts = lf.select(x, y).drop_nulls()
    result = (
        counts.group_by(x_name, y_name)
        .agg(pl.len().cast(pl.UInt64).alias("f12"))
        .with_columns(
            pl.struct(
                pl.col("f12"),
                pl.col("f12").sum().over(x_name).alias("f1"),
                pl.col("f12").sum().over(y_name).alias("f2"),
                pl.col("f12").sum().alias("n"),
            ).alias("freqs")
        )
        .drop("f12")
    )
    return collect_like(result, corpus)


def _apply_measure(
    fn: Measure, f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr
) -> pl.Expr:
    """Compute a measure a caller wrote, and name the column it produces.

    `fn` gets the four counts as Float64 expressions, the same ones the built-in
    measures work from. Its column takes the name `fn` aliased its result with,
    or `fn`'s own name when it aliased nothing. An expression carrying no alias
    reports a name polars supplied for it -- the leftmost count it reads, or
    "literal" when it starts from a constant -- which is what `_DEFAULT_NAMES`
    catches.
    """
    expr = fn(*_as_freqs(f12, f1, f2, n))
    if not isinstance(expr, pl.Expr):
        raise ValueError(
            "a measure must return a polars expression built from the four "
            f"counts it is given, but this one returned {type(expr).__name__}"
        )
    name = expr.meta.output_name(raise_if_undetermined=False)
    if not name or name in _DEFAULT_NAMES:
        name = getattr(fn, "__name__", "")
        if not name or name == "<lambda>":
            raise ValueError(
                "a measure needs a name for the column it produces. Define it "
                "with def, or alias what it returns, e.g. .alias('my_measure')"
            )
    return expr.alias(name)


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
    \\text{PMI}(x,y) = \\log\\frac{P(x,y)}{P(x)\\,P(y)}
                    = \\log\\frac{n\\,f_{12}}{f_1\\,f_2}
    $$

    References
    ----------
    - Church, K. W. and P. Hanks. 1990. Word association norms, mutual information, and
      lexicography. *Computational Linguistics* 16(1): 22-29.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    return ((f12 * n) / (f1 * f2)).log()


def mi3(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute the MI3 association measure for contingency table data.

    Cubes the joint frequency before comparing it to what independence
    predicts, which pulls the ranking away from the rare pairs that dominate
    `pmi` and towards pairs that are both frequent and strongly associated.

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
        A Polars expression that computes the MI3 values for each variable pair.

    Notes
    -----
    MI3 is calculated as:

    $$
    \\text{MI3}(x,y) = \\log\\frac{f_{12}^3}{e_{12}}
                     = \\log\\frac{f_{12}^3\\,n}{f_1\\,f_2}
    $$

    where $e_{12} = f_1\\,f_2 / n$ is the joint frequency expected under
    independence. Like `pmi`, this uses natural logarithms, so the two are on
    the same scale and differ by exactly $2 \\log f_{12}$.

    References
    ----------
    - Daille, B. 1994. *Approche mixte pour l’extraction automatique de
      terminologie: statistiques lexicales et filtres linguistiques.* Ph.D. thesis,
      Université Paris 7.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    return (f12.pow(3) * n / (f1 * f2)).log()


def logdice(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute the log-Dice association measure for contingency table data.

    A logarithmic form of the Dice coefficient, which weighs the joint
    frequency against the two marginals rather than against the corpus size. This
    is the default collocation score in Sketch Engine and #LancsBox.

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
        Grand total (total number of observations). Accepted for a uniform
        signature with the other measures, but not used.

    Returns
    -------
    pl.Expr
        A Polars expression that computes the log-Dice values for each
        variable pair.

    Notes
    -----
    Log-Dice is calculated as:

    $$
    \\text{logDice}(x,y) = 14 + \\log_2\\frac{2\\,f_{12}}{f_1 + f_2}
    $$

    The base-2 logarithm and the constant 14 go together: they place the
    maximum at 14, where every occurrence of one word is an occurrence of the
    pair, and each further point down the scale halves the Dice coefficient.
    Values below 0 are conventionally treated as no association.

    References
    ----------
    - Rychlý, P. 2008. A lexicographer-friendly association score. In
      *Proceedings of Recent Advances in Slavonic Natural Language Processing
      (RASLAN)*, 6-9. Brno: Masaryk University.
    """
    f12, f1, f2, _ = _as_freqs(f12, f1, f2, n)
    return 14 + (2 * f12 / (f1 + f2)).log(2)


def tscore(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute the t-score association measure for contingency table data.

    Scales the gap between the observed joint frequency and the one expected
    under independence by the square root of the observed frequency. Dividing
    by the observed rather than the expected count holds rare pairs down, so
    the ranking favors frequent, dependable pairings -- the complement to what
    `pmi` reports.

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
        A Polars expression that computes the t-score values for each variable
        pair.

    Notes
    -----
    The t-score is calculated as:

    $$
    t(x,y) = \\frac{f_{12} - e_{12}}{\\sqrt{f_{12}}}
    $$

    where $e_{12} = f_1\\,f_2 / n$ is the joint frequency expected under
    independence. It is not a t statistic in the distributional sense -- the
    normality it would need does not hold for word counts -- and is best read
    as a ranking, not a test.

    References
    ----------
    - Church, K. W., W. Gale, P. Hanks, and D. Hindle. 1991. Using statistics
      in lexical analysis. In U. Zernik (ed.), *Lexical Acquisition: Exploiting
      On-Line Resources to Build a Lexicon*, 115-164. Hillsdale: Erlbaum.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    return (f12 - f1 * f2 / n) / f12.sqrt()


def zscore(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Compute the z-score association measure for contingency table data.

    Scales the gap between the observed joint frequency and the one expected
    under independence by the square root of the expected frequency. Dividing
    by the expected count makes it more forgiving of rare pairs than `tscore`,
    and less so than `pmi`.

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
        A Polars expression that computes the z-score values for each variable
        pair.

    Notes
    -----
    The z-score is calculated as:

    $$
    z(x,y) = \\frac{f_{12} - e_{12}}{\\sqrt{e_{12}}}
    $$

    where $e_{12} = f_1\\,f_2 / n$ is the joint frequency expected under
    independence. This is the Poisson form, which takes the variance of the
    count to be its mean.

    References
    ----------
    - Berry-Rogghe, G. L. M. 1973. The computation of collocations and their
      relevance in lexical studies. In A. J. Aitken, R. W. Bailey, and
      N. Hamilton-Smith (eds.), *The Computer and Literary Studies*, 103-112.
      Edinburgh University Press.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    expected = f1 * f2 / n
    return (f12 - expected) / expected.sqrt()


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

    $$
    \\chi^2 = \\frac{n\\,(|n\\,f_{12} - f_1\\,f_2| - c)^2}
                  {f_1\\,f_2\\,(n - f_1)\\,(n - f_2)}
    $$

    where the continuity-correction term is $c = n/2$ when `yates` is True
    and $c = 0$ otherwise.
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

    $$
    \\begin{align*}
    G^2 = 2 \\Bigl(
      &f_{12} \\log\\frac{f_{12}}{e_{12}}
      + (f_1 - f_{12}) \\log\\frac{f_1 - f_{12}}{f_1 - e_{12}}\\\\
      &+ (f_2 - f_{12}) \\log\\frac{f_2 - f_{12}}{f_2 - e_{12}}
      + (n - f_1 - f_2 + f_{12})
        \\log\\frac{n - f_1 - f_2 + f_{12}}{n - f_1 - f_2 + e_{12}}
    \\Bigr)
    \\end{align*}
    $$

    where the four terms are the cells of the 2×2 table and
    $e_{12} = f_1\\,f_2 / n$ is the joint frequency expected under
    independence.

    References
    ----------
    - Dunning, T. 1993. Accurate methods for the statistics of surprise and
    coincidence. *Computational Linguistics* 19:61–74.
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

    $$
    t = \\frac{\\bar{x}_1 - \\bar{x}_2}
             {\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}
    $$

    where $\\bar{x}_i$, $s_i^2$, and $n_i$ are the sample mean, sample
    variance, and sample size of the $i$-th sample.

    The degrees of freedom are approximated using the Welch-Satterthwaite equation:

    $$
    df = \\frac{\\left(\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}\\right)^2}
              {\\frac{(s_1^2/n_1)^2}{n_1 - 1} + \\frac{(s_2^2/n_2)^2}{n_2 - 1}}
    $$

    The effect size is Hedges' g: Cohen's d recovered from the t-statistic, then
    scaled by the small-sample bias correction $J$:

    $$
    d = t\\,\\sqrt{\\frac{2\\left(\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}\\right)}
                      {s_1^2 + s_2^2}}
    \\qquad
    g = J\\,d, \\quad J = 1 - \\frac{3}{4\\,df - 1}
    $$
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

    $$
    \\bar{x}_i = \\frac{s_i}{n_i}
    \\qquad
    \\text{var}_i = \\frac{ss_i - s_i^2 / n_i}{n_i - 1}
    $$

    The test statistic, degrees of freedom, and effect size are then calculated
    using the same formulas as in `welchs_t`, with $\\text{var}_i$ in place of
    $s_i^2$.
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
