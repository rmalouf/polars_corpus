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
    "bic",
    "logratio",
    "pctdiff",
    "oddsratio",
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


def _discounted(count: pl.Expr, discount: float) -> pl.Expr:
    """Stand `discount` in for a count of zero, leaving every other count alone.

    A word missing from one of the two corpora gives the ratio measures a zero
    denominator, and one infinity swamps whatever ranking they are sorted on.
    Nudging only the zeros leaves every other value exactly what the counts say.

    Written as an addition rather than a `when`/`then` so that the expression
    keeps the name of the count it came from. An unnamed `when` reports
    "literal", which two measures in one `with_columns` collide on.
    """
    if not discount:
        return count
    return count + discount * (count == 0).cast(pl.Float64)


def _rel_freqs(
    f12: pl.Expr, f1: pl.Expr, f2: pl.Expr, n: pl.Expr, discount: float
) -> tuple[pl.Expr, pl.Expr]:
    """A word's relative frequency in the target corpus and in the reference.

    Reads the margins as `crosstab(word, corpus_part)` lays them out: `f1` is
    the word's frequency in both corpora together and `f2` is the size of the
    target corpus, so the reference counts are `f1 - f12` out of `n - f2`.
    """
    return (
        _discounted(f12, discount) / f2,
        _discounted(f1 - f12, discount) / (n - f2),
    )


def crosstab(corpus: T_Frame, x: IntoExprColumn, y: IntoExprColumn) -> T_Frame:
    """
    Build the contingency table that the association measures read their counts from.

    Cross-tabulates `x` against `y`: one row per pair of values that occur
    together in `corpus`, holding the four counts of its 2×2 table. Rows
    where either value is null are dropped first.

    Parameters
    ----------
    corpus : DataFrame | LazyFrame
        The frame to cross-tabulate.
    x : IntoExprColumn
        Column name, expression or Series giving the first variable. A Series
        must have the same height as `corpus` and stands in for the column's
        values.
    y : IntoExprColumn
        Column name, expression or Series giving the second variable.

    Returns
    -------
    DataFrame | LazyFrame
        One row per pair of values observed, in no particular order, with
        the columns:

        - the two variable columns, named after `x` and `y`
        - `freqs` : struct with the four counts as fields

            - `f12` : how often the pair occurs
            - `f1` : how often the `x` value occurs, with any `y` value
            - `f2` : how often the `y` value occurs, with any `x` value
            - `n` : the grand total

        Eager if `corpus` is a DataFrame, lazy if it is a LazyFrame.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or
        `x` or `y` does not resolve to a single column of it.
    ShapeError
        If a Series passed for `x` or `y` is not the height of `corpus`.

    Notes
    -----
    The measures in this module read their counts from a `freqs` struct
    like the one here. `keywords` and `collocations` cross-tabulate and
    score in one step.
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
    Measure association with pointwise mutual information (PMI).

    PMI is the log of the observed joint frequency over the frequency
    expected under independence: positive when the pair occurs together
    more often than chance, negative when less. Rare pairs score high,
    and frequent pairs score low.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the PMI of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
    Measure association with MI3.

    Cubes the joint frequency before comparing it to what independence
    predicts, pulling the ranking away from the rare pairs that dominate
    `pmi` towards pairs that are both frequent and strongly associated.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the MI3 of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
    Measure association with log-Dice.

    A logarithmic form of the Dice coefficient, which weighs the joint
    frequency against the two marginals rather than against the corpus size,
    so scores stay comparable across corpora of different sizes. It is the
    default collocation score in Sketch Engine.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations. Accepted for a uniform
        signature with the other measures, but not used.

    Returns
    -------
    pl.Expr
        Expression returning the log-Dice of each pair.

    Notes
    -----
    Log-Dice is calculated as:

    $$
    \\text{logDice}(x,y) = 14 + \\log_2\\frac{2\\,f_{12}}{f_1 + f_2}
    $$

    The base-2 logarithm and the constant 14 go together: they place the
    maximum at 14, where every occurrence of one word is an occurrence of the
    pair, and each further point down the scale halves the Dice coefficient.

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
    Measure association with the t-score.

    Scales the gap between the observed joint frequency and the one expected
    under independence by the square root of the observed frequency. Dividing
    by the observed rather than the expected count holds rare pairs down, so
    the ranking favors frequent, dependable pairings.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the t-score of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
    Measure association with the z-score.

    Scales the gap between the observed joint frequency and the one expected
    under independence by the square root of the expected frequency. For a
    pair that occurs more often than independence predicts the expected
    count is the smaller one, so the same gap scores higher here than under
    `tscore`.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the z-score of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
    Measure association with Pearson's chi-squared (χ²).

    Sums the squared deviations of the four cells of the 2×2 table from what
    independence predicts, each scaled by its expected count. The statistic
    grows with the corpus: the same proportions give a larger value in a
    bigger one, so it is a measure of significance, not of effect size.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.
    yates : bool, default False
        Whether to apply Yates' continuity correction. When True, this matches
        the default behavior of `scipy.stats.chi2_contingency` for 2×2 tables.

    Returns
    -------
    pl.Expr
        Expression returning the chi-squared statistic of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

    Notes
    -----
    For a 2×2 contingency table the statistic reduces to:

    $$
    \\chi^2 = \\frac{n\\,(|n\\,f_{12} - f_1\\,f_2| - c)^2}
                  {f_1\\,f_2\\,(n - f_1)\\,(n - f_2)}
    $$

    where the continuity-correction term is $c = n/2$ when `yates` is True
    and $c = 0$ otherwise.

    References
    ----------
    - Pearson, K. 1900. On the criterion that a given system of deviations
      from the probable in the case of a correlated system of variables is
      such that it can be reasonably supposed to have arisen from random
      sampling. *Philosophical Magazine* 50(302): 157-175.
    - Yates, F. 1934. Contingency tables involving small numbers and the χ²
      test. *Supplement to the Journal of the Royal Statistical Society*
      1(2): 217-235.
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
    Measure association with the log-likelihood ratio (G²).

    Compares the four cells of the 2×2 table with what independence predicts,
    on a log scale. It is the standard keyness statistic: unlike `chisq` it
    stays reliable for rare pairs, where the expected counts are small.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the log-likelihood ratio of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
      coincidence. *Computational Linguistics* 19(1): 61-74.
    """
    return register_plugin_function(
        plugin_path=LIB,
        args=list(_as_freqs(f12, f1, f2, n)),
        function_name="py_loglik",
        is_elementwise=True,
    )


def bic(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    r"""
    Measure association with the Bayes factor BIC.

    Discounts the log-likelihood ratio by a penalty that grows with the corpus
    size, so the threshold a word has to clear rises as the corpora get bigger.
    In a corpus of a hundred million words almost every difference is
    significant by $G^2$; BIC asks instead how much the evidence outweighs the
    number of observations it rests on.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.

    Returns
    -------
    pl.Expr
        Expression returning the BIC of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

    Notes
    -----
    BIC is calculated as:

    $$
    \text{BIC} = |G^2| - \log n
    $$

    where $G^2$ is the log-likelihood ratio and the penalty is $d\log n$ for
    the $d = 1$ degree of freedom of a 2×2 table. Wilson reads the result as
    degrees of evidence: 0-2 is not worth more than a bare mention, 2-6 is
    positive, 6-10 is strong, and above 10 is very strong.

    Wilson writes $G^2$ unsigned. `loglik` here carries the sign of the
    deviation instead, so BIC takes its magnitude and, like `chisq`, says how
    strong the evidence is without saying which way it points. Pair it with
    `logratio` or `pctdiff` for the direction.

    References
    ----------
    - Wilson, A. 2013. Embracing Bayes factors for key item analysis in corpus
      linguistics. In M. Bieswanger and A. Koll-Stobbe (eds.), *New Approaches
      to the Study of Linguistic Variability*, 3-11. Frankfurt: Peter Lang.
    """
    _, _, _, size = _as_freqs(f12, f1, f2, n)
    return loglik(f12, f1, f2, n).abs() - size.log()


def minsens(f12: IntoExpr, f1: IntoExpr, f2: IntoExpr, n: IntoExpr) -> pl.Expr:
    """
    Measure association with minimum sensitivity.

    The smaller of the two conditional probabilities: how often the second
    variable occurs given the first, and how often the first occurs given
    the second.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations. Accepted for a uniform
        signature with the other measures, but not used.

    Returns
    -------
    pl.Expr
        Expression returning the minimum sensitivity of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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
    Measure keyword effect size with Kilgarriff's simple maths metric.

    The ratio of a word's frequency in the target corpus to its frequency in
    the reference corpus. The constant `k` is added to both frequencies, so
    a word missing from one corpus is counted as `k` there rather than as
    zero.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first. Accepted for a uniform signature with the other
        measures, but not used.
    n : IntoExpr
        Grand total: the count of all observations. Accepted for a uniform
        signature with the other measures, but not used.
    k : float
        Smoothing constant added to both the target and reference frequencies.

    Returns
    -------
    pl.Expr
        Expression returning the simple-maths ratio of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

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


def logratio(
    f12: IntoExpr,
    f1: IntoExpr,
    f2: IntoExpr,
    n: IntoExpr,
    discount: float = 0.5,
) -> pl.Expr:
    r"""
    Measure keyword effect size with Hardie's log ratio.

    Compares how common a word is in the target corpus with how common it is in
    the reference, on a base-2 log scale: 1 means twice as common in the target,
    2 means four times, and a negative value means it is rarer there. This is an
    effect size rather than a test. It says how large the difference is, not how
    strong the evidence for it is, and it does not grow with the corpus.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.
    discount : float, default 0.5
        Count to stand in for a frequency of zero, so that a word missing from
        one corpus gets a large log ratio rather than an infinite one. Set it to
        0 to leave the infinities in place.

    Returns
    -------
    pl.Expr
        Expression returning the log ratio of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

    Notes
    -----
    Log ratio is calculated as:

    $$
    \text{LR}(x,y) = \log_2\frac{f_{12} / f_2}{(f_1 - f_{12}) / (n - f_2)}
    $$

    Unlike the other measures here this one is not symmetric in its two
    margins. It reads them as `crosstab` lays them out for a word by corpus
    part table, which is what `keywords` passes it: $f_{12}$ is the word's
    frequency in the target corpus and $f_2$ the size of that corpus, while
    $f_1$ is the word's frequency in both corpora together and $n$ their
    combined size. The reference corpus frequency is therefore $f_1 - f_{12}$
    and its size $n - f_2$.

    A word absent from the reference corpus has no ratio, and `discount`
    replaces that zero frequency. Such a word is then ranked by the constant as
    much as by its own frequency, so the top of a log ratio list is worth
    reading with `min_target_freq` set.

    References
    ----------
    - Hardie, A. 2014. Log Ratio: an informal introduction. *ESRC Centre for
      Corpus Approaches to Social Science (CASS)*, Lancaster University.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    target, reference = _rel_freqs(f12, f1, f2, n, discount)
    return (target / reference).log(2)


def pctdiff(
    f12: IntoExpr,
    f1: IntoExpr,
    f2: IntoExpr,
    n: IntoExpr,
    discount: float = 0.5,
) -> pl.Expr:
    r"""
    Measure keyword effect size as %DIFF.

    Gives the difference between a word's relative frequency in the target
    corpus and in the reference as a percentage of the reference figure: 100
    means twice as common in the target, -50 means half as common. It ranks
    words exactly as `logratio` does, on a percentage scale rather than a
    logarithmic one.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.
    discount : float, default 0.5
        Count to stand in for a frequency of zero, so that a word missing from
        one corpus gets a large percentage rather than an infinite one. Set it
        to 0 to leave the infinities in place.

    Returns
    -------
    pl.Expr
        Expression returning the %DIFF of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

    Notes
    -----
    %DIFF is calculated as:

    $$
    \%\text{DIFF}(x,y) = 100\,
        \frac{f_{12} / f_2 - (f_1 - f_{12}) / (n - f_2)}
             {(f_1 - f_{12}) / (n - f_2)}
    $$

    It reads the two margins as `logratio` does, and the two are the same
    ranking on different scales:
    $\%\text{DIFF} = 100\,(2^{\text{LR}} - 1)$.

    References
    ----------
    - Gabrielatos, C. and A. Marchi. 2011. Keyness: Matching metrics to
      definitions. Paper given at *Corpus Linguistics in the South 1*,
      University of Portsmouth.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    target, reference = _rel_freqs(f12, f1, f2, n, discount)
    # The scaling trails the difference so the expression is named for `f12`
    # rather than for the leading literal; see `_discounted`.
    return (target - reference) / reference * 100


def oddsratio(
    f12: IntoExpr,
    f1: IntoExpr,
    f2: IntoExpr,
    n: IntoExpr,
    discount: float = 0.5,
) -> pl.Expr:
    r"""
    Measure association with the odds ratio.

    Divides the odds that a token of the target corpus is this word by the same
    odds in the reference corpus. 1 means the word is no more likely in one than
    the other, 2 means the odds are twice as high in the target. Like `logratio`
    it measures the size of the difference rather than the evidence for it.

    Parameters
    ----------
    f12 : IntoExpr
        Column name or expression for the joint frequency: how often the two
        variables co-occur. Pass the other three counts the same way.
    f1 : IntoExpr
        Row marginal: how often the first variable occurs, with any value of
        the second.
    f2 : IntoExpr
        Column marginal: how often the second variable occurs, with any
        value of the first.
    n : IntoExpr
        Grand total: the count of all observations.
    discount : float, default 0.5
        Count to stand in for a cell of zero, so that a word missing from one
        corpus gets a large odds ratio rather than an infinite one. Set it to 0
        to leave the infinities in place.

    Returns
    -------
    pl.Expr
        Expression returning the odds ratio of each pair.

    Raises
    ------
    ValueError
        If any count is not a column name or a Polars expression; a Series
        is not accepted here.

    Notes
    -----
    For a 2×2 contingency table the odds ratio is:

    $$
    \text{OR} = \frac{o_{11}\,o_{22}}{o_{12}\,o_{21}}
              = \frac{f_{12}\,(n - f_1 - f_2 + f_{12})}
                     {(f_1 - f_{12})\,(f_2 - f_{12})}
    $$

    Swapping the two margins leaves this unchanged, so unlike `logratio` it does
    not care which of them is the corpus part. Values run from 0 to infinity
    with 1 at independence, which is an awkward scale to average or plot. Take
    the logarithm for one that is symmetric about 0.

    Odds and relative frequency are close for anything that is a small share of
    a corpus, so for most words the odds ratio comes out near
    $2^{\text{LR}}$. It rises above that for a word frequent enough that
    removing its own tokens changes the corpus it is measured against.

    Replacing a zero cell with `discount` is the usual 0.5 correction for an
    odds ratio with an empty cell.

    References
    ----------
    - Pojanapunya, P. and R. Watson Todd. 2018. Log-likelihood and odds ratio:
      Keyness statistics for different purposes of keyword analysis. *Corpus
      Linguistics and Linguistic Theory* 14(1): 133-167.
    """
    f12, f1, f2, n = _as_freqs(f12, f1, f2, n)
    o11 = _discounted(f12, discount)
    o12 = _discounted(f1 - f12, discount)
    o21 = _discounted(f2 - f12, discount)
    o22 = _discounted(n - f1 - f2 + f12, discount)
    return (o11 * o22) / (o12 * o21)


def welchs_t(x1: IntoExprColumn, x2: IntoExprColumn, alt: str = "twosided") -> pl.Expr:
    """
    Compare the means of two independent samples with Welch's t-test.

    A form of Student's t-test that does not assume the two samples have the
    same variance. It is computed once over the two columns as a whole, or
    once per group when used in a `group_by` aggregation.

    Parameters
    ----------
    x1 : IntoExprColumn
        Column name or expression giving the first sample. Nulls are left
        out of the sample rather than counted in its size.
    x2 : IntoExprColumn
        Column name or expression giving the second sample; nulls are left
        out of it too.
    alt : {'greater', 'less', 'twosided'}, default 'twosided'
        Alternative hypothesis to test:

        - 'twosided' : the means are unequal (two-tailed test)
        - 'greater' : the mean of x1 is greater than the mean of x2 (one-tailed)
        - 'less' : the mean of x1 is less than the mean of x2 (one-tailed)

    Returns
    -------
    pl.Expr
        Expression returning a struct with the test as its fields:

        - 't' : float
            The t-statistic of the test
        - 'p' : float
            The p-value of the test
        - 'df' : float
            The degrees of freedom used in the test
        - 'g' : float
            Hedges' g, a measure of effect size

        All four fields come out null when the test cannot be performed:
        a sample with fewer than two values, or zero variance in both.

    Raises
    ------
    ValueError
        If `alt` is not one of 'twosided', 'greater', or 'less'.

    See Also
    --------
    welchs_t_from_stats : The same test from summary statistics, one test
        per row.

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

    References
    ----------
    - Lijffijt, J., T. Nevalainen, T. Säily, P. Papapetrou, K. Puolamäki,
      and H. Mannila. 2016. Significance testing of word frequencies in corpora.
      *Digital Scholarship in the Humanities* 31(2): 374-397.
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
    Run Welch's t-test from summary statistics.

    The same test as `welchs_t`, computed from sums, sums of squares and
    sample sizes rather than raw samples. Each row of the six statistics
    gives one test.

    Parameters
    ----------
    s1 : IntoExprColumn
        Column name or expression for the sum of the first sample (its mean
        times its size works too). The other five statistics are column
        names or expressions in the same way.
    ss1 : IntoExprColumn
        Sum of squares of the first sample.
    n1 : IntoExprColumn
        Sample size of the first sample.
    s2 : IntoExprColumn
        Sum of the second sample.
    ss2 : IntoExprColumn
        Sum of squares of the second sample.
    n2 : IntoExprColumn
        Sample size of the second sample.
    alt : {'greater', 'less', 'twosided'}, default 'twosided'
        Alternative hypothesis to test:

        - 'twosided' : the means are unequal (two-tailed test)
        - 'greater' : the mean of the first sample is greater than the second (one-tailed)
        - 'less' : the mean of the first sample is less than the second (one-tailed)

    Returns
    -------
    pl.Expr
        Expression returning, for each row, a struct with the test as its
        fields:

        - 't' : float
            The t-statistic of the test
        - 'p' : float
            The p-value of the test
        - 'df' : float
            The degrees of freedom used in the test
        - 'g' : float
            Hedges' g, a measure of effect size

        All four fields come out null when the test cannot be performed:
        a sample with fewer than two values, or zero variance in both.

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
