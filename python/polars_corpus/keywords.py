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


def keywords(
    target: T_Frame,
    reference: T_Frame,
    term: IntoExprColumn,
    method: str,
    min_target_tf: int = 0,
    min_target_df: int = 0,
) -> pl.DataFrame:
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
    method : {'ttest', 'pmi', 'll'}
        Association measure used to rank keywords:

        - 'ttest' : Welch's t-test on per-file relative frequencies
        - 'pmi' : Pointwise Mutual Information
        - 'll' : Log-likelihood ratio (G²)
    min_target_tf : int, default 0
        Minimum term frequency in the target corpus required for a word to be
        included in the results. Ignored when `method` is 'ttest'.
    min_target_df : int, default 0
        Minimum document frequency in the target corpus required for a word to be
        included in the results. Ignored when `method` is 'ttest'.

    Returns
    -------
    pl.DataFrame
        Keywords ranked by association strength, most target-specific first.

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

    target = target.lazy()
    reference = reference.lazy()

    combined = pl.concat(
        [
            target.with_columns(pl.lit("target").alias(PART_FIELD)),
            reference.with_columns(pl.lit("reference").alias(PART_FIELD)),
        ]
    )

    if method == "ttest":
        result = keywords_ttest(combined, term, min_target_tf)
    else:
        term_name = term if isinstance(term, str) else term.meta.output_name()
        combined = combined.with_columns(term)
        freq_table = plc.crosstab(combined, term_name, PART_FIELD)
        target_df = (
            combined.filter(pl.col(PART_FIELD) == "target")
            .group_by(term_name)
            .agg(pl.col("file_id").n_unique().alias("target_df"))
        )
        freq_table = freq_table.join(target_df, on=term_name, how="left")
        result = keywords_assoc(freq_table, method, min_target_tf, min_target_df)

    return result.collect()


def keywords_assoc(
    freq_table: pl.LazyFrame, method: str, min_target_tf: int, min_target_df: int
) -> pl.LazyFrame:
    """
    Rank keywords from a crosstab frequency table using PMI or log-likelihood.

    Parameters
    ----------
    freq_table : pl.LazyFrame
        Crosstab of word by corpus part (see `polars_corpus.crosstab`), with a
        `freqs` struct column and a `PART_FIELD` column identifying target vs.
        reference rows.
    method : {'pmi', 'll'}
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
        If `method` is not 'pmi' or 'll'.
    """
    match method:
        case "pmi":
            assoc_expr = pl.col("freqs").corpus.pmi().alias("PMI")
        case "ll":
            assoc_expr = pl.col("freqs").corpus.loglik().alias("LogLik")
        case _:
            raise ValueError()

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


# #%%
# ttest
# #%%
# male = (
#     ttest.filter(pl.col("stat") > 0)
#     .select("norm", "stat", "pval")
#     .join(lex, on="norm", how="left")
#     .head(25)
#     .select(pl.all().name.suffix("_m"))
# )
#


# # import sys
# #
# # if ".." not in sys.path:
# #     sys.path.append("..")
#
# import html
#
# import great_tables as gt
# import numpy as np
# import polars as pl
# from scipy.stats import chi2_contingency, fisher_exact
#
# import polars_corpus as plc
# #%%
# bnc = pl.scan_parquet("bnc.parquet")
# speakers = pl.scan_parquet("bnc-speakers.parquet")
#
# speakers = speakers.filter(pl.col("sex").is_in(["m", "f"]))
# speakers.group_by("sex").len()
#
# bnc = (
#     bnc.filter(pl.col("text_type") == "CONVRSN")
#     .join(speakers, on="speaker_id", how="inner")
#     .with_columns(pl.col("token").str.to_lowercase().alias("norm"))
# )
# #%%
# table = (
#     bnc.group_by((pl.col("norm") == "husband").alias("is_husband"), "sex")
#     .agg(pl.col("norm").len().alias("count"))
#     .with_columns(rel_freq=pl.col("count") / pl.col("count").sum().over("sex"))
#     .sort(by=["sex", "is_husband"])
# ).collect()
# table
# #%%
# table = np.array(table["count"]).reshape(2, 2)
# table
# #%%
# fisher_exact(table)
# #%%
# chi2_contingency(table)
# #%%
# np.sqrt(chi2_contingency(table).statistic / table.sum())
# #%%
# table = (
#     bnc.group_by((pl.col("norm") == "wife").alias("is_wife"), "sex")
#     .agg(pl.col("norm").len().alias("count"))
#     .with_columns(rel_freq=pl.col("count") / pl.col("count").sum().over("sex"))
#     .sort(by=["sex", "is_wife"])
# ).collect()
# table = np.array(table["count"]).reshape(2, 2)
# table
# #%%
# chi2_contingency(table)
# #%%
# np.sqrt(chi2_contingency(table).statistic / table.sum())
# #%%
# np.sqrt(chi2_contingency(table).statistic / table.sum())
# #%%
# fisher_exact(table)
# #%%
# table = (
#     bnc.group_by((pl.col("norm") == "is").alias("is_is"), "sex")
#     .agg(pl.col("norm").len().alias("count"))
#     .with_columns(rel_freq=pl.col("count") / pl.col("count").sum().over("sex"))
#     .sort(by=["sex", "is_is"])
# ).collect()
# table = np.array(table["count"]).reshape(2, 2)
# table
# #%%
# chi2_contingency(table)
# #%%
# np.sqrt(chi2_contingency(table).statistic / table.sum())
# #%%
# fisher_exact(table)
# #%% md
# -----
# #%% md
# Stefanowitsch (2020), pp. 378–380
# #%%
# freq_table = plc.crosstab(bnc, "norm", "sex")
# #%%
# ll = (
#     freq_table.with_columns(LL=pl.col("freqs").corpus.loglik())
#     .sort(by="LL", descending=True)
#     .collect()
# )
# #%%
# f12 = pl.col("freqs").struct.field("f12")
#
# lex = ll.select("norm", "sex", f12.alias("f12")).pivot(on="sex", index="norm")
#
# male = (
#     ll.filter(
#         pl.col("sex") == "m",
#         )
#     .select("norm", "LL")
#     .join(lex, on="norm", how="left")
#     .head(25)
#     .select(pl.all().name.suffix("_m"))
# )
#
# female = (
#     ll.filter(
#         pl.col("sex") == "f",
#         )
#     .select("norm", "LL")
#     .join(lex, on="norm", how="left")
#     .head(25)
# )
#
# tbl = (
#     pl.concat([male, female], how="horizontal")
#     .with_columns(pl.lit("").alias("spacer"))
#     .select("norm_m", "f_m", "m_m", "LL_m", "spacer", "norm", "f", "m", "LL")
#     .style.fmt_number(["LL_m", "LL"], decimals=2)
#     .fmt_integer(["f_m", "m_m", "f", "m"], use_seps=True)
#     .fmt(html.escape, columns=["norm_m", "norm"])
#     .tab_spanner("MALE", ["norm_m", "f_m", "m_m", "LL_m"])
#     .tab_spanner("FEMALE", ["norm", "f", "m", "LL"])
#     .cols_label(
#         {
#             "norm_m": "word",
#             "LL_m": "LL",
#             "norm": "word",
#             "spacer": "",
#             "f_m": "f freq",
#             "m_m": "m freq",
#             "f": "f freq",
#             "m": "m freq",
#         }
#     )
#     .tab_style(
#         style=gt.style.css("width:50px"), locations=gt.loc.body(columns=["spacer"])
#     )
#     .opt_row_striping()
#     .opt_vertical_padding(0.6)
# )
#
# tbl.save("LL")
# #%% md
# Lijffijt et al. (2016)
# #%%
# bnc = bnc.collect()
#
# n_m = bnc.filter(pl.col("sex") == "m").n_unique("file_id")
# n_f = bnc.filter(pl.col("sex") == "f").n_unique("file_id")
#
# ttest = (
#     bnc.group_by("file_id", "sex", "norm")
#     .agg(pl.len().alias("freq"))
#     .with_columns(n=pl.sum("freq").over(["file_id", "sex"]))
#     .with_columns(
#         (pl.col("freq") / pl.sum("freq").over(["file_id", "sex"])).alias("rel_freq")
#     )
#     .group_by("sex", "norm")
#     .agg(s=pl.col("rel_freq").sum(), ss=(pl.col("rel_freq") * pl.col("rel_freq")).sum())
#     .pivot(on="sex", index=["norm"])
#     .drop_nulls()
#     .with_columns(n_m=n_m, n_f=n_f)
#     .with_columns(plc.welchs_t_from_stats("s_m", "ss_m", "n_m", "s_f", "ss_f", "n_f"))
#     .unnest("t_test")
#     .sort(by="pval")
# )
# #%%
# ttest
# #%%
# male = (
#     ttest.filter(pl.col("stat") > 0)
#     .select("norm", "stat", "pval")
#     .join(lex, on="norm", how="left")
#     .head(25)
#     .select(pl.all().name.suffix("_m"))
# )
#
# male
#
# female = (
#     ttest.filter(pl.col("stat") < 0)
#     .select("norm", "stat", "pval")
#     .join(lex, on="norm", how="left")
#     .head(25)
# )
#
# tbl = (
#     pl.concat([male, female], how="horizontal")
#     .with_columns(pl.lit("").alias("spacer"))
#     .select(
#         "norm_m",
#         "f_m",
#         "m_m",
#         "stat_m",
#         "pval_m",
#         "spacer",
#         "norm",
#         "f",
#         "m",
#         "stat",
#         "pval",
#     )
#     .style.fmt_number(["stat_m", "stat"], decimals=2)
#     .fmt_number(["pval_m", "pval"], decimals=4)
#     .fmt_integer(["f_m", "m_m", "f", "m"], use_seps=True)
#     .fmt(html.escape, columns=["norm_m", "norm"])
#     .tab_spanner("MALE", ["norm_m", "f_m", "m_m", "stat_m", "pval_m"])
#     .tab_spanner("FEMALE", ["norm", "f", "m", "stat", "pval"])
#     .cols_label(
#         {
#             "norm_m": "word",
#             "stat_m": "t",
#             "pval_m": "p",
#             "stat": "t",
#             "pval": "p",
#             "norm": "word",
#             "spacer": "",
#             "f_m": "f freq",
#             "m_m": "m freq",
#             "f": "f freq",
#             "m": "m freq",
#         }
#     )
#     .tab_style(
#         style=gt.style.css("width:50px"), locations=gt.loc.body(columns=["spacer"])
#     )
#     .opt_row_striping()
#     .opt_vertical_padding(0.6)
# )
#
# tbl.save("ttest")
