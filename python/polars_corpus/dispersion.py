from __future__ import annotations

import polars as pl

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_columns,
    check_expr,
    collect_like,
)

__all__ = [
    "dispersion",
]

RANGE_METHODS = ("range", "range%")
SD_METHODS = ("sd", "cv", "cv%", "d")
METHODS = RANGE_METHODS + SD_METHODS + ("da", "dp")

# - Juilland, A., & Chang-Rodriguez, E. (1964). Frequency dictionary of Spanish words.
#   The Hague: Mouton.
# - Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
#   International Journal of Corpus Linguistics, 13(4), 403-437.
# - Burch, B., Egbert, J., & Biber, D. (2017). Measuring and interpreting lexical
#   dispersion in corpus linguistics. Journal of Research Design and Statistics in
#   Linguistics and Communication Science, 3(2), 189-216.


def dispersion(
    corpus: T_Frame,
    expr: IntoExpr,
    method: str,
    file_id_column: str = "file_id",
) -> T_Frame:
    """
    Measure how evenly each word is spread across the files of a corpus.

    Parameters
    ----------
    corpus : DataFrame | LazyFrame
        Corpus to measure dispersion in.
    expr : IntoExpr
        Column name or expression identifying the word/type to measure
        (e.g. token or lemma).
    method : {'range', 'range%', 'sd', 'cv', 'cv%', 'd', 'da', 'dp'}
        Dispersion measure to compute:

        - 'range' : number of files the word occurs in
        - 'range%' : the range as a percentage of the files in the corpus
        - 'sd' : standard deviation of the per-file frequencies
        - 'cv' : coefficient of variation, `sd` over the mean
        - 'cv%' : the coefficient of variation as a percentage
        - 'd' : Julliand's D, `1 - cv / sqrt(N - 1)` for `N` files
        - 'da' : Burch's DA, from the average difference between pairs of files
        - 'dp' : Gries's DP, how far the word's spread over the files falls
          from the corpus's own
    file_id_column : str, default "file_id"
        Column holding file ids, defining the parts the word is spread across.

    Returns
    -------
    T_Frame
        One row per word with its dispersion, in no particular order.
        Eager if `corpus` is a DataFrame, lazy if it is a LazyFrame.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.dispersion(corpus, "lemma", "d")

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
        missing a column `dispersion` needs; if `expr` is not a column name or
        expression; or if `method` is not one of the measures listed above.

    Notes
    -----
    Only the columns `expr` and `file_id_column` name are read.

    'range', 'range%' and 'sd' scale with a word's frequency, so they are
    comparable only between words of similar frequency; the rest divide that
    scale out. 'range%' does divide out the number of files, so unlike 'range'
    it can be compared across corpora cut into different numbers of parts. 'd'
    and 'da' both run from 0 (the word falls in a single file) to 1 (spread
    perfectly evenly), but 'da' compares the files to each other rather than to
    their mean, so one outlying file sways it less. 'dp' runs the other way,
    from 0 (spread exactly as the corpus is) up towards 1, and reaches that
    ceiling only for a word confined to a vanishingly small file.

    Sort the result to rank words, keeping in mind which end is which: 'sd',
    'cv', 'cv%' and 'dp' measure unevenness, so an even spread is the low end,
    while 'range', 'range%', 'd' and 'da' measure evenness directly and an even
    spread is the high end.

    A corpus of a single file gives NaN for 'sd', 'cv', 'cv%', 'd' and 'da':
    dispersion across one part is undefined. 'dp' gives 0 instead, there being
    no second file for the word to be spread unevenly over.
    """
    method = check_choice(method, METHODS)
    term = as_expr(expr)
    lf = as_corpus(corpus)

    check_columns(lf, [file_id_column], param="file_id_column")
    term_name = check_expr(lf, term)

    if method in RANGE_METHODS:
        files = pl.col(file_id_column).n_unique()
        counts = (
            lf.select(term, file_id_column)
            .with_columns(files.alias("_N"))
            .group_by(term_name)
            .agg(files.alias("range"), pl.col("_N").first())
        )
        measure = {
            "range": pl.col("range"),
            "range%": (100 * pl.col("range") / pl.col("_N")).alias("range%"),
        }[method]
        return collect_like(counts.select(term_name, measure), corpus)

    # DP weighs each file's share of the word against a share of its own, so it
    # works from the counts and the file sizes side by side rather than from the
    # per-file frequencies the measures below share.
    if method == "dp":
        result = dispersion_dp(lf, term, term_name, file_id_column)
        return collect_like(result, corpus)

    per_file = (
        lf.group_by(term, file_id_column)
        .agg(pl.len().alias("_n"))
        .with_columns(pl.col(file_id_column).n_unique().alias("_N"))
    )
    sizes = lf.group_by(file_id_column).agg(pl.len().alias("_size"))
    per_file = per_file.join(sizes, on=file_id_column).select(
        term_name, "_N", (pl.col("_n") / pl.col("_size")).alias("_n")
    )

    if method in SD_METHODS:
        result = dispersion_sd(per_file, term_name, method)
    else:
        result = dispersion_da(per_file, term_name)

    return collect_like(result, corpus)


def dispersion_sd(
    per_file: pl.LazyFrame,
    term_name: str,
    method: str,
) -> pl.LazyFrame:
    """Measure dispersion by how much the per-file frequencies vary.

    `per_file` holds one row per word per file it occurs in, with the frequency
    in `_n` and the corpus-wide file count in `_N`.
    """
    # Files a word never occurs in add nothing to either sum; `_N` restores them.
    stats = per_file.group_by(term_name).agg(
        pl.col("_n").sum().alias("_S"),
        (pl.col("_n") ** 2).sum().alias("_Q"),
        pl.col("_N").first(),
    )

    mean = pl.col("_S") / pl.col("_N")
    # Population sd: the files are the whole corpus, not a sample drawn from it.
    variance = (pl.col("_Q") - pl.col("_S") ** 2 / pl.col("_N")) / pl.col("_N")
    # Watch out for rounding, which can drive variance below 0.
    sd = variance.clip(lower_bound=0).sqrt()
    cv = sd / mean
    cv_norm = (cv / (pl.col("_N") - 1).sqrt()).clip(upper_bound=1)

    measure = {
        "sd": sd.alias("sd"),
        "cv": cv.alias("cv"),
        "cv%": (cv * 100).alias("cv%"),
        "d": (1 - cv_norm).alias("D"),
    }[method]

    return stats.select(term_name, measure)


def dispersion_da(per_file: pl.LazyFrame, term_name: str) -> pl.LazyFrame:
    """Measure dispersion by how much the per-file frequencies differ pairwise.

    `per_file` holds one row per word per file it occurs in, with the frequency
    in `_n` and the corpus-wide file count in `_N`.
    """
    # Notes on the O(n log n) method used here:
    #   https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/gmd.htm
    #   https://bshlgrs.github.io/2016/12/29/gini.html
    weight = 2 * pl.int_range(pl.len()) - 2 * pl.len() + pl.col("_N").first() + 1
    stats = per_file.group_by(term_name).agg(
        (pl.col("_n").sort() * weight).sum().alias("_P"),
        pl.col("_n").sum().alias("_S"),
        pl.col("_N").first(),
    )
    da = 1 - pl.col("_P") / ((pl.col("_N") - 1) * pl.col("_S"))

    return stats.select(term_name, da.alias("DA"))


def dispersion_dp(
    lf: pl.LazyFrame,
    term: pl.Expr,
    term_name: str,
    file_id_column: str,
) -> pl.LazyFrame:
    """Measure dispersion by how far the word's spread falls from the corpus's.

    Half the total gap between the share of the word each file holds and the
    share of the corpus it is expected to hold.
    """
    counts = lf.group_by(term, file_id_column).agg(pl.len().alias("_n"))
    expected = pl.col("_size") / pl.col("_size").sum()
    sizes = (
        lf.group_by(file_id_column)
        .agg(pl.len().alias("_size"))
        .select(file_id_column, expected.alias("_e"))
    )

    observed = pl.col("_n") / pl.col("_n").sum()
    # A file the word is absent from is off by its whole expected share, and
    # those shares sum to 1 over the corpus, so dropping the present files'
    # shares from the total leaves exactly what the absent ones contribute.
    stats = (
        counts.join(sizes, on=file_id_column)
        .group_by(term_name)
        .agg(((observed - pl.col("_e")).abs() - pl.col("_e")).sum().alias("_A"))
    )

    dp = ((1 + pl.col("_A")) / 2).clip(lower_bound=0)

    return stats.select(term_name, dp.alias("DP"))
