from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl
import polars_corpus as plc
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

from ._typing import T_Frame

__all__ = [
    "keywords",
]

PART_FIELD = "_part"

# Scott, M. (1997). PC analysis of key words—and key key words. System, 25(2), 233-245.
# Kilgarriff, A. (2009, July). Simple maths for keywords. In Proceedings of the Corpus Linguistics Conference. Liverpool, UK.
# Leech, G., & Fallon, R. (1992). Computer corpora: What do they tell us about culture? ICAMEJournal,16,29–50.

# TODO: Add Gries's (2001) KL divergence method?


def keywords(
    target: T_Frame,
    reference: T_Frame,
    term: IntoExprColumn,
    method: str,
    min_target_tf: int = 0,
    min_target_df: int = 0,
    k: Optional[int] = None,
) -> T_Frame:
    """
    Identify keywords by comparing frequencies in a target corpus against a reference corpus.

    Parameters
    ----------
    target : DataFrame | LazyFrame
        Target corpus (DataFrame or LazyFrame) whose keywords are being extracted.
    reference : DataFrame | LazyFrame
        Reference corpus (DataFrame or LazyFrame) that `target` is compared against.
    term : IntoExprColumn
        Column identifying the word/type to compute keyness for (e.g. token or lemma).
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
    k: int, default None
        Constant to be add in Kilgarriff's "simple maths parameter". Only used
        if `method` is 'smp'.

    Returns
    -------
    T_Frame
        Keywords ranked by association strength, most target-specific first.
        Eager if `target` is a DataFrame, lazy if it is a LazyFrame.

    Raises
    ------
    ValueError
        If `target` or `reference` is not a Polars DataFrame or LazyFrame.
    """
    if not isinstance(target, pl.DataFrame) and not isinstance(target, pl.LazyFrame):
        raise ValueError()
    if not isinstance(reference, pl.DataFrame) and not isinstance(
        reference, pl.LazyFrame
    ):
        raise ValueError()

    eager = isinstance(target, pl.DataFrame)
    target = target.lazy()
    reference = reference.lazy()
    if isinstance(term, str):
        term = pl.col(term)

    combined = pl.concat(
        [
            target.with_columns(pl.lit("target").alias(PART_FIELD)),
            reference.with_columns(pl.lit("reference").alias(PART_FIELD)),
        ]
    )

    if method == "ttest":
        result = keywords_ttest(combined, term, min_target_tf)
    else:
        term_name = term.meta.output_name()
        combined = combined.with_columns(term)
        freq_table = plc.crosstab(combined, term_name, PART_FIELD)
        target_df = (
            combined.filter(pl.col(PART_FIELD) == "target")
            .group_by(term_name)
            .agg(pl.col("file_id").n_unique().alias("target_df"))
        )
        freq_table = freq_table.join(target_df, on=term_name, how="left")
        result = keywords_assoc(freq_table, method, min_target_tf, min_target_df, k)

    return result.collect() if eager else result


def keywords_assoc(
    freq_table: pl.LazyFrame,
    method: str,
    min_target_tf: int,
    min_target_df: int,
    k: Optional[int],
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


    Returns
    -------
    pl.LazyFrame
        Target-corpus rows sorted by association strength, descending.

    Raises
    ------
    ValueError
        If `method` is not 'pmi', 'll', 'chisq', 'smp', or 'minsens'.
    """
    match method:
        case "pmi":
            assoc_expr = pl.col("freqs").corpus.pmi().alias("PMI")
        case "ll":
            assoc_expr = pl.col("freqs").corpus.loglik().alias("LogLik")
        case "chisq":
            assoc_expr = pl.col("freqs").corpus.chisq().alias("ChiSq")
        case "minsens":
            assoc_expr = pl.col("freqs").corpus.minsens().alias("MinSens")
        case "smp":
            if k is None:
                raise ValueError("k is required for smp")
            assoc_expr = pl.col("freqs").corpus.smp(k).alias("SMP")
        case _:
            raise ValueError(f"Unknown method {method}")

    result = (
        freq_table.filter(
            pl.col("freqs").struct.field("f12") >= min_target_tf,
            pl.col("target_df") >= min_target_df,
            pl.col(PART_FIELD) == "target",
        )
        .with_columns(assoc_expr)
        .select(pl.exclude(PART_FIELD))
        .sort(by=assoc_expr.meta.output_name(), descending=True)
    )
    return result


def keywords_ttest(
    combined: pl.LazyFrame, term: IntoExprColumn, min_freq: int
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
    term : IntoExprColumn
        Column identifying the word/type to compute keyness for.
    min_freq : int
        Unused; present for API symmetry with `keywords_assoc`.

    Returns
    -------
    pl.LazyFrame
        Words with a higher mean relative frequency in the target corpus
        (`stat` > 0), sorted by p-value ascending.
    """
    term_name = term.meta.output_name()

    result = (
        combined.group_by("file_id", PART_FIELD, term)
        .agg(pl.len().alias("freq"))
        .with_columns(
            rel_freq=pl.col("freq") / pl.sum("freq").over("file_id", PART_FIELD),
            n=pl.col("file_id").n_unique().over(PART_FIELD),
        )
        .group_by(PART_FIELD, term_name)
        .agg(
            n=pl.first("n"),
            s=pl.col("rel_freq").sum(),
            ss=(pl.col("rel_freq") ** 2).sum(),
        )
        .pivot(on=PART_FIELD, on_columns=["target", "reference"], index=term_name)
        .drop_nulls()
        .with_columns(
            plc.welchs_t_from_stats(
                "s_target",
                "ss_target",
                "n_target",
                "s_reference",
                "ss_reference",
                "n_reference",
            )
        )
        .select(term_name, "t_test")
        .unnest("t_test")
        .filter(pl.col("stat") > 0)
        .sort(by="pval")
    )
    return result
