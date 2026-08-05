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
SD_METHODS = ("sd", "cv", "cv%", "d")

METHODS = SD_METHODS

# Julliand, A., & Chang-Rodriguez, E. (1964). Frequency dictionary of Spanish words.
# The Hague: Mouton.
# Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
# International Journal of Corpus Linguistics, 13(4), 403-437.

# TODO: Add Gries's DP, Burch's DA


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
    method : {'sd', 'cv', 'cv%', 'd'}
        Dispersion measure to compute:

        - 'sd' : standard deviation of the per-file frequencies
        - 'cv' : coefficient of variation, `sd` over the mean
        - 'cv%' : the coefficient of variation as a percentage
        - 'd' : Julliand's D, `1 - cv / sqrt(N - 1)` for `N` files
    normalize : bool, default True
        Measure the per-file relative frequencies rather than the raw counts.
        Leave this on unless the files are all the same length: with raw counts
        a word looks unevenly spread simply because the files it falls in differ
        in size.
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

    'sd' scales with a word's frequency, so it is comparable only between words
    of similar frequency; 'cv', 'cv%' and 'd' divide that scale out. 'd' runs
    from 0 (the word falls in a single file) to 1 (spread perfectly evenly).

    Sort the result to rank words, keeping in mind which end is which: 'sd',
    'cv' and 'cv%' measure variation, so an even spread is the low end, while
    'd' measures evenness directly and an even spread is the high end.

    A corpus of a single file gives NaN: dispersion across one part is undefined.
    """
    method = check_choice(method, METHODS)
    term = as_expr(expr)
    lf = as_corpus(corpus)

    # Check up front, so column errors don't surface from inside a query plan.
    check_columns(lf, [file_id_column], param="file_id_column")
    term_name = check_expr(lf, term)

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
        raise ValueError(f"Unknown method {method!r}")

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


# def dispersion_da(per_file: pl.LazyFrame,
#                   term_name: str) -> pl.LazyFrame:
#     """Compute Burch's DA dispersion metric."""
#
#     stats = (
#         per_file.group_by(term_name)
#         .agg(pl.col("_n").sum().alias("_N"))
