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
    drop_null_rows,
)

__all__ = [
    "keywords",
]

PART_FIELD = "_part"

METHODS = ("ttest", "pmi", "ll", "chisq", "smp", "minsens")

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
        for (e.g., token or lemma). Note that `expr` is evaluated against the
        concatenated target+reference corpora, not either alone.
    method : {'chisq', 'll', 'minsens', 'pmi', 'smp', 'ttest'}
        [Association metric](assoc.md) used to rank keywords:
        - 'chisq' : Pearson's chi-squared (χ²)
        - 'll' : Log-likelihood ratio (G²)
        - 'minsens' : Minimum sensitivity
        - 'pmi' : Pointwise Mutual Information
        - 'smp' : Kilgarriff's simple maths parameter (requires `k`)
        - 'ttest' : Welch's t-test on per-file relative frequencies
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

        Every method but 'ttest' returns the frequency table with one column
        named for the measure. 'ttest' returns the words more frequent in the
        target, ranked by p-value ascending, with columns `t`, `p`, `df`, and
        `g`. The test statistic `t` and the p-value `p` indicate the strength of
        evidence for an association, while Hedges' `g` is the effect size.

    Raises
    ------
    ValueError
        If `target` or `reference` is not a Polars DataFrame or LazyFrame, is
        empty, or is missing a column `keywords` needs; if `expr` is not a
        column name or expression; if `method` is not one of the measures
        listed above; or if `method` is 'smp' and `k` is missing or not
        positive.

    Warns
    -----
    UserWarning
        If rows are dropped from either corpus for holding a null. Raised only
        for an eager corpus: counting the dropped rows of a LazyFrame would mean
        reading it before the caller has asked for anything.

    Notes
    -----
    Only the columns `expr` and `file_id_column` name are read, so the target
    and reference corpora need not have matching schemas otherwise. Rows holding a null
    in either are dropped.

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
    >>> plc.keywords(target, reference, "lemma", "ll", min_target_df=5)
    >>> plc.keywords(target, reference, "token", "ttest", file_id_column="text_id")
    """
    method = check_choice(method, METHODS)
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
    if method == "ttest" and (min_target_tf or min_target_df):
        warnings.warn(
            "min_target_tf and min_target_df are not applied when method='ttest'",
            stacklevel=2,
        )

    target_lf = as_corpus(target, "target corpus")
    reference_lf = as_corpus(reference, "reference corpus")

    # Check up front, so column errors don't surface from inside a query plan.
    parts = (("target", target_lf, target), ("reference", reference_lf, reference))
    names = []
    for part, corpus, _ in parts:
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

    # Select the term itself, so differently annotated corpora still concatenate,
    # then drop the rows holding a null in either column. A token with no value
    # for `expr` is not an occurrence of anything, and one with no file id is in
    # no document; left in, they pad the corpus totals the measures divide by and
    # add a file of their own to the document counts.
    selected = []
    for part, corpus, source in parts:
        rows = drop_null_rows(
            corpus.select(keyword_expr, file_id_column), source, f"{part} corpus"
        )
        selected.append(rows.with_columns(pl.lit(part).alias(PART_FIELD)))
    combined = pl.concat(selected)

    if method == "ttest":
        result = _keywords_ttest(combined, expr_name, file_id_column)
    else:
        freq_table = crosstab(combined, expr_name, PART_FIELD)
        target_df = (
            combined.filter(pl.col(PART_FIELD) == "target")
            .group_by(expr_name)
            .agg(pl.col(file_id_column).n_unique().alias("target_df"))
        )
        freq_table = freq_table.join(target_df, on=expr_name, how="left")
        result = _keywords_assoc(freq_table, method, min_target_tf, min_target_df, k)

    return collect_like(result, target)


def _keywords_assoc(
    freq_table: pl.LazyFrame,
    method: str,
    min_target_tf: int,
    min_target_df: int,
    k: Optional[float],
) -> pl.LazyFrame:
    """Rank keywords from a crosstab frequency table by association strength.

    `freq_table` is a crosstab of word by corpus part (see
    `polars_corpus.crosstab`), with a `freqs` struct and a `PART_FIELD` column.
    Returns the target-corpus rows, sorted descending.
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


def _keywords_ttest(
    combined: pl.LazyFrame,
    expr_name: str,
    file_id_column: str = "file_id",
) -> pl.LazyFrame:
    """Rank keywords by Welch's t-test on per-file relative frequencies.

    `combined` is the two corpora concatenated, with a `PART_FIELD` column
    marking which is which. Returns the words overrepresented in the target
    (`t` > 0), sorted by p-value ascending, with the test's `t`, `p`, `df`, and
    `g` (Hedges' g) columns unnested alongside the word.
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
        .filter(pl.col("t") > 0)
        .sort(by="p")
    )
    return result
