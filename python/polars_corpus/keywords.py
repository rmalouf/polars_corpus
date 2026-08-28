from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Optional

import polars as pl

from ._typing import IntoExpr, Measure, T_Frame
from .assoc import (
    _apply_measure,
    bic,
    chisq,
    crosstab,
    loglik,
    logratio,
    mi3,
    minsens,
    oddsratio,
    pctdiff,
    pmi,
    smp,
    tscore,
    welchs_t_from_stats,
    zscore,
)
from .utils import (
    as_corpus,
    as_expr,
    check_columns,
    check_measure,
    check_expr,
    collect_like,
)

__all__ = [
    "keywords",
]

PART_FIELD = "_part"

# The measures on offer that are functions of the four counts, each with the
# column it is reported in. 'ttest' is not one of them: it works from the
# per-file relative frequencies rather than the corpus totals.
MEASURES: dict[str, tuple[Callable[..., pl.Expr], str]] = {
    "pmi": (pmi, "PMI"),
    "mi3": (mi3, "MI3"),
    "ll": (loglik, "LogLik"),
    "bic": (bic, "BIC"),
    "chisq": (chisq, "ChiSq"),
    "tscore": (tscore, "TScore"),
    "zscore": (zscore, "ZScore"),
    "minsens": (minsens, "MinSens"),
    "smp": (smp, "SMP"),
    "logratio": (logratio, "LogRatio"),
    "pctdiff": (pctdiff, "%DIFF"),
    "oddsratio": (oddsratio, "OddsRatio"),
}

METHODS = ("ttest", *MEASURES)

# TODO: Add Gries's (2001) KL divergence method?


def keywords(
    target: T_Frame,
    reference: T_Frame,
    expr: IntoExpr,
    method: str | Measure,
    min_target_freq: int = 0,
    min_target_range: int = 0,
    k: Optional[float] = None,
    file_id_column: str = "file_id",
) -> T_Frame:
    """
    Identify keywords by comparing frequencies in a target corpus against a reference corpus.

    Parameters
    ----------
    target : DataFrame | LazyFrame
        Target corpus (DataFrame or LazyFrame) whose keywords are being extracted.
    reference : DataFrame | LazyFrame
        Reference corpus (DataFrame or LazyFrame) that `target` is compared against.
    expr : IntoExpr
        Column name or expression identifying the word/type to compute keyness
        for (e.g., token or lemma). Note that `expr` is evaluated against the
        combined target+reference corpora.
    method : str | callable
        [Association metric](assoc.md) used to rank keywords:

         - 'bic' : Bayes factor BIC, log-likelihood penalized by corpus size
         - 'chisq' : Pearson's chi-squared (χ²)
         - 'll' : Log-likelihood ratio (G²)
         - 'logratio' : Hardie's log ratio, the effect size (column `LogRatio`)
         - 'mi3' : MI3, which pulls the ranking back towards frequent words
         - 'minsens' : Minimum sensitivity
         - 'oddsratio' : Odds ratio, the effect size (column `OddsRatio`)
         - 'pctdiff' : %DIFF, the effect size in percent (column `%DIFF`)
         - 'pmi' : Pointwise Mutual Information, which favors rare words
         - 'smp' : Kilgarriff's simple maths parameter (requires `k`)
         - 'tscore' : t-score, which favors frequent words
         - 'ttest' : Welch's t-test on per-file relative frequencies
         - 'zscore' : z-score

         'll', 'chisq' and 'bic' measure the evidence that a word's two
         frequencies differ, which grows with the size of the corpora.
         'logratio', 'oddsratio' and 'pctdiff' measure how large that
         difference is, which does not. The literature expects one of each,
         because a large corpus makes a tiny difference significant.

         `method` can also be a `Callable` which takes the four counts `f12`, `f1`, `f2`
         and `n` described under `Returns`. It receives them as Polars expressions
         and returns one expression.
    min_target_freq : int, default 0
        Minimum frequency in the target corpus required for a word to be
        included in the results.
    min_target_range : int, default 0
        Minimum range in the target corpus -- the number of distinct files a
        word must occur in -- required for it to be included in the results.
    k: float, default None
        Constant added to both frequencies in Kilgarriff's "simple maths
        parameter"; larger values favor more frequent words. Required when
        `method` is 'smp' and unused otherwise.
    file_id_column : str, default "file_id"
        Column holding file ids, used for range counts and for the
        per-file relative frequencies underlying 'ttest'.

    Returns
    -------
    DataFrame | LazyFrame
        Keywords ranked by association strength, most target-specific first.
        Eager if `target` is a DataFrame, lazy if it is a LazyFrame.

        Every method but 'ttest' returns the frequency table with one column
        named for the measure. 'ttest' returns the words more frequent in the
        target, ranked by p-value ascending, with the target-corpus counts the
        thresholds are applied to (`target_freq`, `target_range`) and the columns
        `t`, `p`, `df`, and `g`. The test statistic `t` and the p-value `p`
        indicate the strength of evidence for an association, while Hedges' `g`
        is the effect size. Note that `df` here is the test's degrees of
        freedom; a word's range is reported as `target_range`.

    Raises
    ------
    ValueError
        If `target` or `reference` is not a Polars DataFrame or LazyFrame, is
        empty, or is missing a column `keywords` needs; if `expr` is not a
        column name or expression; if `method` is not one of the measures
        listed above or a function; if a measure of your own returns something
        that is not an expression, or has no name to give its column; or if
        `method` is 'smp' and `k` is missing or not positive.

    Notes
    -----
    Rows with null values in either `expr` or `file_id_column` are dropped.

    'bic' and 'chisq' are unsigned, so a word much rarer in the target corpus
    than in the reference ranks alongside one much commoner. Every other
    measure is signed, and puts the words the target overuses at the top.

    'logratio', 'oddsratio' and 'pctdiff' have no value for a word absent from
    the reference corpus, and stand a count of 0.5 in for that zero. The
    ranking among such words then rests on that constant as much as on the
    data, which `min_target_freq` is the way to keep in hand.

    References
    ----------
    - Hofland, K. and Johansson, S. 1982. *Word frequencies in British and
      American English.* Norwegian Computing Centre for the Humanities. Bergen.
    - Leech, G. and R. Fallon. 1992. Computer corpora – What do they tell us about
      culture? *ICAME Journal* 16: 29–50.
    - Lijffijt, J., T. Nevalainen, T. Säily, P. Papapetrou, K. Puolamäki, and
      H. Mannila. 2016. Significance testing of word frequencies in corpora. *Digital
      Scholarship in the Humanities* 31(2): 374-397.
    - Scott, M. 1997. PC analysis of key words—and key key words. *System* 25(2): 233-245.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.keywords(target, reference, "lemma", "ll", min_target_range=5)
    >>> plc.keywords(target, reference, "token", "ttest", file_id_column="text_id")
    >>> # An effect size beside the significance measure, on the same words:
    >>> keys = plc.keywords(target, reference, "lemma", "ll", min_target_freq=10)
    >>> keys.with_columns(pl.col("freqs").corpus.logratio().alias("LogRatio"))
    >>> # A measure of your own -- Hofland and Johansson's difference coefficient:
    >>> def diff_coefficient(f12, f1, f2, n):
    ...     target_rf, reference_rf = f12 / f2, (f1 - f12) / (n - f2)
    ...     return (target_rf - reference_rf) / (target_rf + reference_rf)
    >>> plc.keywords(target, reference, "lemma", diff_coefficient, min_target_freq=10)
    """
    method = check_measure(method, METHODS)
    keyword_expr = as_expr(expr)
    if method == "smp":
        if k is None:
            raise ValueError(
                "method 'smp' needs a value for k: pass k=100 for Kilgarriff's "
                "default, or a larger value to favor more frequent words"
            )
        if k <= 0:
            raise ValueError(f"k must be a positive number, got {k}")
    elif k is not None:
        warnings.warn(
            f"k={k} is only used when method='smp'; ignoring it", stacklevel=2
        )

    target_lf = as_corpus(target, "target corpus")
    reference_lf = as_corpus(reference, "reference corpus")

    selected = []
    for part, corpus in (("target", target_lf), ("reference", reference_lf)):
        check_columns(
            corpus, [file_id_column], f"{part} corpus", param="file_id_column"
        )
        check_expr(corpus, keyword_expr, f"{part} corpus")
        rows = corpus.select(keyword_expr, file_id_column).drop_nulls()
        selected.append(rows.with_columns(pl.lit(part).alias(PART_FIELD)))
    combined = pl.concat(selected)

    keyword_name = combined.collect_schema().names()[0]

    if method == "ttest":
        result = _keywords_ttest(
            combined, keyword_name, min_target_freq, min_target_range, file_id_column
        )
    else:
        freq_table = crosstab(combined, keyword_name, PART_FIELD)
        target_range = (
            combined.filter(pl.col(PART_FIELD) == "target")
            .group_by(keyword_name)
            .agg(pl.col(file_id_column).n_unique().alias("target_range"))
        )
        freq_table = freq_table.join(target_range, on=keyword_name, how="left")
        result = _keywords_assoc(
            freq_table, method, min_target_freq, min_target_range, k
        )

    return collect_like(result, target)


def _keywords_assoc(
    freq_table: pl.LazyFrame,
    method: str | Measure,
    min_target_freq: int,
    min_target_range: int,
    k: Optional[float],
) -> pl.LazyFrame:
    """Rank keywords from a crosstab frequency table by association strength.

    `freq_table` is a crosstab of word by corpus part (see
    `polars_corpus.crosstab`), with a `freqs` struct and a `PART_FIELD` column.
    Returns the target-corpus rows, sorted descending.
    """
    freqs = pl.col("freqs")
    f12 = freqs.struct.field("f12")
    f1 = freqs.struct.field("f1")
    f2 = freqs.struct.field("f2")
    n = freqs.struct.field("n")

    if isinstance(method, str):
        measure, name = MEASURES[method]
        # 'smp' is the one built-in that takes more than the four counts.
        counts = (f12, f1, f2, n, k) if method == "smp" else (f12, f1, f2, n)
        assoc_expr = measure(*counts).alias(name)
    else:
        assoc_expr = _apply_measure(method, f12, f1, f2, n)

    result = (
        freq_table.filter(
            f12 >= min_target_freq,
            pl.col("target_range") >= min_target_range,
            pl.col(PART_FIELD) == "target",
        )
        .with_columns(assoc_expr)
        .select(pl.exclude(PART_FIELD))
        .sort(by=assoc_expr.meta.output_name(), descending=True)
    )
    return result


def _keywords_ttest(
    combined: pl.LazyFrame,
    expr_name: str,
    min_target_freq: int,
    min_target_range: int,
    file_id_column: str = "file_id",
) -> pl.LazyFrame:
    """Rank keywords by Welch's t-test on per-file relative frequencies.

    `combined` is the two corpora concatenated, with a `PART_FIELD` column
    marking which is which. Returns the words overrepresented in the target
    (`t` > 0) that clear `min_target_freq` and `min_target_range`, sorted by p-value
    ascending, with the target counts and the test's `t`, `p`, `df`, and `g`
    (Hedges' g) columns alongside the word.
    """
    target_counts = (
        combined.filter(pl.col(PART_FIELD) == "target")
        .group_by(expr_name)
        .agg(
            target_freq=pl.len(),
            target_range=pl.col(file_id_column).n_unique(),
        )
    )

    result = (
        combined.group_by(file_id_column, PART_FIELD, expr_name)
        .agg(pl.len().alias("freq"))
        .with_columns(
            rel_freq=pl.col("freq") / pl.sum("freq").over(file_id_column, PART_FIELD),
            n=pl.col(file_id_column).n_unique().over(PART_FIELD),
        )
        .group_by(PART_FIELD, expr_name)
        .agg(
            n=pl.first("n"),
            s=pl.col("rel_freq").sum(),
            ss=(pl.col("rel_freq") ** 2).sum(),
        )
        .pivot(on=PART_FIELD, on_columns=["target", "reference"], index=expr_name)
        .with_columns(
            pl.col("s_target", "ss_target", "s_reference", "ss_reference").fill_null(0),
            pl.col("n_target", "n_reference").fill_null(strategy="max"),
        )
        .with_columns(
            welchs_t_from_stats(
                "s_target",
                "ss_target",
                "n_target",
                "s_reference",
                "ss_reference",
                "n_reference",
            )
        )
        .select(expr_name, "t_test")
        .unnest("t_test")
        .join(target_counts, on=expr_name, how="inner")
        .filter(
            pl.col("t") > 0,
            pl.col("target_freq") >= min_target_freq,
            pl.col("target_range") >= min_target_range,
        )
        .select(expr_name, "target_freq", "target_range", "t", "p", "df", "g")
        .sort(by="p")
    )
    return result
