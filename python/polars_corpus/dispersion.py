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

# Grouped by family: each shares an aggregation and gets its own branch below.
RANGE_METHODS = ("range", "range%")

SD_METHODS = ("sd", "cv", "cv%", "d")

METHODS = RANGE_METHODS + SD_METHODS + ("da",)

# Julliand, A., & Chang-Rodriguez, E. (1964). Frequency dictionary of Spanish words.
# The Hague: Mouton.
# Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
# International Journal of Corpus Linguistics, 13(4), 403-437.
# Burch, B., Egbert, J., & Biber, D. (2017). Measuring and interpreting lexical
# dispersion in corpus linguistics. Journal of Research Design and Statistics in
# Linguistics and Communication Science, 3(2), 189-216.

# TODO: Add Gries's DP


def dispersion(
    corpus: T_Frame,
    expr: IntoExpr,
    method: str,
    normalize: bool = True,
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
    method : {'range', 'range%', 'sd', 'cv', 'cv%', 'd', 'da'}
        Dispersion measure to compute:

        - 'range' : number of files the word occurs in
        - 'range%' : the range as a percentage of the files in the corpus
        - 'sd' : standard deviation of the per-file frequencies
        - 'cv' : coefficient of variation, `sd` over the mean
        - 'cv%' : the coefficient of variation as a percentage
        - 'd' : Julliand's D, `1 - cv / sqrt(N - 1)` for `N` files
        - 'da' : Burch's DA, from the average difference between pairs of files
    normalize : bool, default True
        Measure the per-file relative frequencies rather than the raw counts.
        Leave this on unless the files are all the same length: with raw counts
        a word looks unevenly spread simply because the files it falls in differ
        in size. Ignored by 'range' and 'range%', which count files rather than
        tokens.
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
    >>> # Raw counts, for a corpus cut into equal-sized parts:
    >>> plc.dispersion(corpus, "lemma", "cv", normalize=False)

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
    comparable only between words of similar frequency; 'cv', 'cv%', 'd' and
    'da' divide that scale out. 'range%' does divide out the number of files,
    so unlike 'range' it can be compared across corpora cut into different
    numbers of parts. 'd' and 'da' both run from 0 (the word falls in a single
    file) to 1 (spread perfectly evenly), but 'da' compares the files to each
    other rather than to their mean, so one outlying file sways it less.

    Sort the result to rank words, keeping in mind which end is which: 'sd',
    'cv' and 'cv%' measure variation, so an even spread is the low end, while
    'range', 'range%', 'd' and 'da' measure evenness directly and an even
    spread is the high end.

    A corpus of a single file gives NaN for every measure but 'range' and
    'range%': dispersion across one part is undefined.
    """
    method = check_choice(method, METHODS)
    term = as_expr(expr)
    lf = as_corpus(corpus)

    # Check up front, so column errors don't surface from inside a query plan.
    check_columns(lf, [file_id_column], param="file_id_column")
    term_name = check_expr(lf, term)

    # Range only asks which files a word falls in, not how often, so it needs
    # neither the per-file frequencies below nor `normalize`.
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

    # Broadcast the file count rather than cross-joining it in: every row of the
    # corpus lands in some group here, so the file ids are all still present.
    per_file = (
        lf.group_by(term, file_id_column)
        .agg(pl.len().alias("_n"))
        .with_columns(pl.col(file_id_column).n_unique().alias("_N"))
    )
    if normalize:
        sizes = lf.group_by(file_id_column).agg(pl.len().alias("_size"))
        per_file = per_file.join(sizes, on=file_id_column).select(
            term_name, "_N", (pl.col("_n") / pl.col("_size")).alias("_n")
        )
    else:
        # Float64 up front: the sum of squares below overflows a count dtype.
        per_file = per_file.select(term_name, "_N", pl.col("_n").cast(pl.Float64))

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
    # Rounding again: a word in a single file overshoots the ceiling of 1.
    cv_norm = (cv / (pl.col("_N") - 1).sqrt()).clip(upper_bound=1)

    # Choose the expression we need to evaluate
    measure = {
        "sd": sd.alias("sd"),
        "cv": cv.alias("cv"),
        "cv%": (cv * 100).alias("cv%"),
        "d": (1 - cv_norm).alias("D"),
    }[method]

    return stats.select(term_name, measure)


# DA is 1 - Gini, though Burch et al. give it as an O(n^2) loop over the pairs
# and never draw the connection. Gini's mean difference, and the O(n log n) form
# of it that the sorted weights below come from:
# https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/gmd.htm
# https://bshlgrs.github.io/2016/12/29/gini.html
def dispersion_da(per_file: pl.LazyFrame, term_name: str) -> pl.LazyFrame:
    """Measure dispersion by how much the per-file frequencies differ pairwise.

    `per_file` holds one row per word per file it occurs in, with the frequency
    in `_n` and the corpus-wide file count in `_N`.
    """
    # Sorted, the sum of |ni - nj| over every pair of files weights each
    # frequency by how many fall below it less how many above: `2i - _N + 1` at
    # rank `i`. The files a word is absent from are zeros, so they sort to the
    # front and contribute nothing themselves -- they only shift the ranks of
    # the rest by `_N - len`, which folds into the weight below.
    weight = 2 * pl.int_range(pl.len()) - 2 * pl.len() + pl.col("_N").first() + 1
    stats = per_file.group_by(term_name).agg(
        (pl.col("_n").sort() * weight).sum().alias("_P"),
        pl.col("_n").sum().alias("_S"),
        pl.col("_N").first(),
    )

    # Scaling that sum by the pair count and by twice the mean leaves the count
    # of files cancelled out: DA = 1 - (_P / (_N(_N-1)/2)) / (2 * _S / _N).
    da = 1 - pl.col("_P") / ((pl.col("_N") - 1) * pl.col("_S"))

    return stats.select(term_name, da.alias("DA"))
