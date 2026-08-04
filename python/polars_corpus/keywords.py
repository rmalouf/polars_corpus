from __future__ import annotations

import warnings
from typing import Optional

import polars as pl

from ._typing import IntoExpr, T_Frame
from .assoc import chisq, crosstab, loglik, minsens, pmi, smp, welchs_t_from_stats
from .utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_columns,
    check_expr,
    collect_like,
)

__all__ = [
    "keywords",
]

PART_FIELD = "_part"

METHODS = ("ttest", "pmi", "ll", "chisq", "smp", "minsens")

# Scott, M. (1997). PC analysis of key words—and key key words. System, 25(2), 233-245.
# Kilgarriff, A. (2009, July). Simple maths for keywords. In Proceedings of the Corpus Linguistics Conference. Liverpool, UK.
# Leech, G., & Fallon, R. (1992). Computer corpora: What do they tell us about culture? ICAMEJournal,16,29–50.

# TODO: Add Gries's (2001) KL divergence method?


def keywords(
    target: T_Frame,
    reference: T_Frame,
    expr: IntoExpr,
    method: str,
    min_target_tf: int = 0,
    min_target_df: int = 0,
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
        for (e.g. token or lemma). A Series is not accepted: `expr` is evaluated
        against the concatenated target+reference corpora, not either alone.
    method : {'ttest', 'pmi', 'll', 'chisq', 'smp', 'minsens'}
        Association measure used to rank keywords:

        - 'ttest' : Welch's t-test on per-file relative frequencies
        - 'pmi' : Pointwise Mutual Information
        - 'll' : Log-likelihood ratio (G²)
        - 'chisq' : Pearson's chi-squared (χ²)
        - 'smp' : Kilgarriff's simple maths parameter (requires `k`)
        - 'minsens' : Minimum sensitivity
    min_target_tf : int, default 0
        Minimum term frequency in the target corpus required for a word to be
        included in the results. Ignored when `method` is 'ttest'.
    min_target_df : int, default 0
        Minimum document frequency in the target corpus required for a word to be
        included in the results. Ignored when `method` is 'ttest'.
    k: float, default None
        Constant added to both frequencies in Kilgarriff's "simple maths
        parameter"; larger values favor more frequent words. Required when
        `method` is 'smp' and unused otherwise.
    file_id_column : str, default "file_id"
        Column holding file ids, used for document frequencies and for the
        per-file relative frequencies underlying 'ttest'.

    Returns
    -------
    T_Frame
        Keywords ranked by association strength, most target-specific first.
        Eager if `target` is a DataFrame, lazy if it is a LazyFrame.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.keywords(target, reference, "lemma", "ll", min_target_df=5)
    >>> # A corpus whose file ids live in another column:
    >>> plc.keywords(target, reference, "lemma", "ttest", file_id_column="text_id")

    Raises
    ------
    ValueError
        If `target` or `reference` is not a Polars DataFrame or LazyFrame, is
        empty, or is missing a column `keywords` needs; if `expr` is not a
        column name or expression; if `method` is not one of the measures
        listed above; or if `method` is 'smp' and `k` is missing or not
        positive.

    Notes
    -----
    Only the columns `expr` and `file_id_column` name are read, so the target
    and reference corpora need not have matching schemas otherwise.
    """
    method = check_choice(method, METHODS)
    keyword_expr = as_expr(
        expr,
        hint=" It is evaluated over the target and reference corpora together,"
        " so a Series taken from one of them would not line up.",
    )
    if method == "smp":
        if k is None:
            raise ValueError(
                "method 'smp' needs a value for k: pass k=1 for Kilgarriff's "
                "default, or a larger value to favor more frequent words"
            )
        if k <= 0:
            raise ValueError(f"k must be a positive number, got {k}")
    elif k is not None:
        warnings.warn(
            f"k={k} is only used when method='smp'; ignoring it", stacklevel=2
        )
    if method == "ttest" and (min_target_tf or min_target_df):
        warnings.warn(
            "min_target_tf and min_target_df are not applied when method='ttest'",
            stacklevel=2,
        )

    target_lf = as_corpus(target, "target corpus")
    reference_lf = as_corpus(reference, "reference corpus")

    # Resolving the term against each schema keeps the errors about missing
    # columns in this function rather than deep in a query plan.
    parts = (("target", target_lf), ("reference", reference_lf))
    names = []
    for part, corpus in parts:
        check_columns(
            corpus, [file_id_column], f"{part} corpus", param="file_id_column"
        )
        names.append(check_expr(corpus, keyword_expr, f"{part} corpus"))
    if names[0] != names[1]:
        raise ValueError(
            f"expr names a different column in each corpus: {names[0]!r} in the "
            f"target corpus but {names[1]!r} in the reference corpus"
        )
    expr_name = names[0]

    # Selecting the term itself rather than the columns behind it keeps corpora
    # with different annotation columns concatenable, whatever shape `expr` takes.
    combined = pl.concat(
        [
            corpus.select(keyword_expr, file_id_column).with_columns(
                pl.lit(part).alias(PART_FIELD)
            )
            for part, corpus in parts
        ]
    )

    if method == "ttest":
        result = keywords_ttest(combined, expr_name, file_id_column)
    else:
        freq_table = crosstab(combined, expr_name, PART_FIELD)
        target_df = (
            combined.filter(pl.col(PART_FIELD) == "target")
            .group_by(expr_name)
            .agg(pl.col(file_id_column).n_unique().alias("target_df"))
        )
        freq_table = freq_table.join(target_df, on=expr_name, how="left")
        result = keywords_assoc(freq_table, method, min_target_tf, min_target_df, k)

    return collect_like(result, target)


def keywords_assoc(
    freq_table: pl.LazyFrame,
    method: str,
    min_target_tf: int,
    min_target_df: int,
    k: Optional[float],
) -> pl.LazyFrame:
    """
    Rank keywords from a crosstab frequency table using PMI or log-likelihood.

    Parameters
    ----------
    freq_table : pl.LazyFrame
        Crosstab of word by corpus part (see `polars_corpus.crosstab`), with a
        `freqs` struct column and a `PART_FIELD` column identifying target vs.
        reference rows.
    method : {'pmi', 'll', 'chisq', 'smp', 'minsens'}
        Association measure to compute.
    min_target_tf : int
        Minimum term frequency in the target corpus required for a word to
        be included.
    min_target_df : int
        Minimum document frequency in the target corpus required for a word to
        be included.
    k : float, optional
        Smoothing constant for 'smp'; required by that method only.

    Returns
    -------
    pl.LazyFrame
        Target-corpus rows sorted by association strength, descending.
    """
    # Call the measures directly rather than through the `.corpus` namespace,
    # which is registered at runtime and so is invisible to type checkers.
    freqs = pl.col("freqs")
    f12 = freqs.struct.field("f12")
    f1 = freqs.struct.field("f1")
    f2 = freqs.struct.field("f2")
    n = freqs.struct.field("n")

    match method:
        case "pmi":
            assoc_expr = pmi(f12, f1, f2, n).alias("PMI")
        case "ll":
            assoc_expr = loglik(f12, f1, f2, n).alias("LogLik")
        case "chisq":
            assoc_expr = chisq(f12, f1, f2, n).alias("ChiSq")
        case "minsens":
            assoc_expr = minsens(f12, f1, f2, n).alias("MinSens")
        case "smp":
            # keywords() has already rejected a missing k.
            assert k is not None
            assoc_expr = smp(f12, f1, f2, n, k).alias("SMP")
        case _:
            raise ValueError(f"Unknown method {method!r}")

    result = (
        freq_table.filter(
            f12 >= min_target_tf,
            pl.col("target_df") >= min_target_df,
            pl.col(PART_FIELD) == "target",
        )
        .with_columns(assoc_expr)
        .select(pl.exclude(PART_FIELD))
        .sort(by=assoc_expr.meta.output_name(), descending=True)
    )
    return result


def keywords_ttest(
    combined: pl.LazyFrame,
    expr_name: str,
    file_id_column: str = "file_id",
) -> pl.LazyFrame:
    """
    Rank keywords by Welch's t-test on per-file relative frequencies.

    Computes, for each word, the relative frequency in every file of the
    combined target+reference corpus, then compares the target and reference
    distributions of those per-file relative frequencies with Welch's t-test.

    Parameters
    ----------
    combined : pl.LazyFrame
        Target and reference corpora concatenated, with a `file_id` column and
        a `PART_FIELD` column identifying target vs. reference rows.
    expr_name : str
        Column identifying the word/type to compute keyness for.
    file_id_column : str, default "file_id"
        Column holding file ids, defining the units the t-test compares.

    Returns
    -------
    pl.LazyFrame
        Words with a higher mean relative frequency in the target corpus
        (`stat` > 0), sorted by p-value ascending.
    """
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
        .drop_nulls()
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
        .filter(pl.col("stat") > 0)
        .sort(by="pval")
    )
    return result
