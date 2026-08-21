from __future__ import annotations

import polars as pl

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_choices,
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

# The column each measure is reported in.
COLUMNS = {
    "range": "range",
    "range%": "range%",
    "sd": "sd",
    "cv": "cv",
    "cv%": "cv%",
    "d": "D",
    "da": "DA",
    "dp": "DP",
}

# - Juilland, A., & Chang-Rodriguez, E. (1964). Frequency dictionary of Spanish words.
#   The Hague: Mouton.
# - Gries, S. Th. (2008). Dispersions and adjusted frequencies in corpora.
#   International Journal of Corpus Linguistics, 13(4), 403-437.
# - Burch, B., Egbert, J., & Biber, D. (2017). Measuring and interpreting lexical
#   dispersion in corpus linguistics. Journal of Research Design and Statistics in
#   Linguistics and Communication Science, 3(2), 189-216.

# TODO: more measures, if they turn out to be wanted
#  -- Carroll's D2, Rosengren's S, Gries's KLD, DPnorm, and the adjusted
#  frequencies (U, AF) that pair with # them.


def dispersion(
    corpus: T_Frame,
    expr: IntoExpr,
    method: str | list[str],
    min_freq: int = 0,
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
    method : str | list of str
        Dispersion measure to compute, or a list of them to compute together:

        - 'range' : number of files the word occurs in
        - 'range%' : the range as a percentage of the files in the corpus
        - 'sd' : standard deviation of the per-file frequencies
        - 'cv' : coefficient of variation, `sd` over the mean
        - 'cv%' : the coefficient of variation as a percentage
        - 'd' : Julliand's D, `1 - cv / sqrt(N - 1)` for `N` files
        - 'da' : Burch's DA, from the average difference between pairs of files
        - 'dp' : Gries's DP, how far the word's spread over the files falls
          from the corpus's own
    min_freq : int, default 0
        Minimum corpus frequency a word needs to be reported. Every measure
        here is unstable for rare words, so raising this is usually the first
        thing to do with a ranked result.
    file_id_column : str, default "file_id"
        Column holding file ids, defining the parts the word is spread across.

    Returns
    -------
    T_Frame
        One row per word, in no particular order, with its corpus frequency in
        `freq` and one column per measure asked for, in the order asked for:
        'range', 'range%', 'sd', 'cv', 'cv%', 'D', 'DA' or 'DP'. Eager if
        `corpus` is a DataFrame, lazy if it is a LazyFrame.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
        missing a column `dispersion` needs; if `expr` is not a column name or
        expression; or if `method` is not one of the measures listed above, or
        a list of them.

    Notes
    -----
    Rows holding a null in either `expr` and `file_id_column` are dropped.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.dispersion(corpus, "lemma", "d")
    >>> # Rank the reasonably frequent words from least to most evenly spread:
    >>> plc.dispersion(corpus, "lemma", "d", min_freq=50).sort("D")
    >>> # Several measures side by side, to see where they disagree:
    >>> plc.dispersion(corpus, "lemma", ["range", "d", "da", "dp"], min_freq=50)
    """
    methods = check_choices(method, METHODS)
    term = as_expr(expr)
    lf = as_corpus(corpus)

    check_columns(lf, [file_id_column], param="file_id_column")
    term_name = check_expr(lf, term)

    lf = lf.select(term, file_id_column).drop_nulls()

    range_methods = [m for m in methods if m in RANGE_METHODS]
    sd_methods = [m for m in methods if m in SD_METHODS]
    frames = []

    if range_methods:
        frames.append(_dispersion_range(lf, term_name, file_id_column, range_methods))

    if sd_methods or "da" in methods:
        per_file = (
            lf.group_by(term_name, file_id_column)
            .agg(pl.len().alias("_n"))
            .with_columns(pl.col(file_id_column).n_unique().alias("_N"))
        )
        sizes = lf.group_by(file_id_column).agg(pl.len().alias("_size"))
        per_file = per_file.join(sizes, on=file_id_column).select(
            term_name,
            "_N",
            pl.col("_n").alias("_f"),
            (pl.col("_n") / pl.col("_size")).alias("_n"),
        )
        if sd_methods:
            frames.append(_dispersion_sd(per_file, term_name, sd_methods))
        if "da" in methods:
            frames.append(_dispersion_da(per_file, term_name))

    if "dp" in methods:
        frames.append(_dispersion_dp(lf, term_name, file_id_column))

    result = frames[0]
    for frame in frames[1:]:
        # Every group reports `freq`, and every group agrees about it.
        result = result.join(frame.drop("freq"), on=term_name)

    if min_freq:
        result = result.filter(pl.col("freq") >= min_freq)

    result = result.select(term_name, "freq", *(COLUMNS[m] for m in methods))

    return collect_like(result, corpus)


def _dispersion_range(
    lf: pl.LazyFrame,
    term_name: str,
    file_id_column: str,
    methods: list[str],
) -> pl.LazyFrame:
    """Measure dispersion by how many of the files the word occurs in at all.

    `lf` holds one row per token, with the word in `term_name`.
    """
    files = pl.col(file_id_column).n_unique()
    counts = (
        lf.with_columns(files.alias("_N"))
        .group_by(term_name)
        .agg(files.alias("range"), pl.len().alias("freq"), pl.col("_N").first())
    )
    measures = {
        "range": pl.col("range"),
        "range%": (100 * pl.col("range") / pl.col("_N")).alias("range%"),
    }

    return counts.select(term_name, "freq", *(measures[m] for m in methods))


def _dispersion_sd(
    per_file: pl.LazyFrame,
    term_name: str,
    methods: list[str],
) -> pl.LazyFrame:
    """Measure dispersion by how much the per-file frequencies vary.

    `per_file` holds one row per word per file it occurs in: relative frequency
    in `_n`, raw count in `_f`, corpus-wide file count in `_N`.
    """
    # Files a word never occurs in add nothing to either sum; `_N` restores them.
    stats = per_file.group_by(term_name).agg(
        pl.col("_n").sum().alias("_S"),
        (pl.col("_n") ** 2).sum().alias("_Q"),
        pl.col("_f").sum().alias("freq"),
        pl.col("_N").first(),
    )

    mean = pl.col("_S") / pl.col("_N")
    variance = (pl.col("_Q") - pl.col("_S") ** 2 / pl.col("_N")) / pl.col("_N")
    sd = variance.clip(lower_bound=0).sqrt()
    cv = sd / mean
    cv_norm = (cv / (pl.col("_N") - 1).sqrt()).clip(upper_bound=1)

    measures = {
        "sd": sd.alias("sd"),
        "cv": cv.alias("cv"),
        "cv%": (cv * 100).alias("cv%"),
        "d": (1 - cv_norm).alias("D"),
    }

    return stats.select(term_name, "freq", *(measures[m] for m in methods))


def _dispersion_da(per_file: pl.LazyFrame, term_name: str) -> pl.LazyFrame:
    """Measure dispersion by how much the per-file frequencies differ pairwise.

    `per_file` is as in `_dispersion_sd`.
    """
    # Notes on the O(n log n) method used here:
    #   https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/gmd.htm
    #   https://bshlgrs.github.io/2016/12/29/gini.html
    weight = 2 * pl.int_range(pl.len()) - 2 * pl.len() + pl.col("_N").first() + 1
    stats = per_file.group_by(term_name).agg(
        (pl.col("_n").sort() * weight).sum().alias("_P"),
        pl.col("_n").sum().alias("_S"),
        pl.col("_f").sum().alias("freq"),
        pl.col("_N").first(),
    )
    da = 1 - pl.col("_P") / ((pl.col("_N") - 1) * pl.col("_S"))

    return stats.select(term_name, "freq", da.alias("DA"))


def _dispersion_dp(
    lf: pl.LazyFrame,
    term_name: str,
    file_id_column: str,
) -> pl.LazyFrame:
    """Measure dispersion by how far the word's spread falls from the corpus's.

    Half the total gap between the share of the word each file holds and the
    share of the corpus it is expected to hold.
    """
    counts = lf.group_by(term_name, file_id_column).agg(pl.len().alias("_n"))
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
        .agg(
            ((observed - pl.col("_e")).abs() - pl.col("_e")).sum().alias("_A"),
            pl.col("_n").sum().alias("freq"),
        )
    )

    dp = ((1 + pl.col("_A")) / 2).clip(lower_bound=0)

    return stats.select(term_name, "freq", dp.alias("DP"))
