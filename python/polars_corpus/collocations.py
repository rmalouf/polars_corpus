from __future__ import annotations

from typing import Optional

import polars as pl

from ._typing import IntoExpr
from .assoc import chisq, logdice, loglik, mi3, minsens, pmi, tscore, zscore
from .search import LazySearchResults, SearchResults, _check_count, _window_span
from .utils import as_expr, check_choices, check_expr

__all__ = [
    "collocations",
]

METHODS = (
    "freq",
    "pmi",
    "mi3",
    "logdice",
    "ll",
    "chisq",
    "tscore",
    "zscore",
    "minsens",
)

# The column each measure is reported in.
COLUMNS = {
    "freq": "freq",
    "pmi": "PMI",
    "mi3": "MI3",
    "logdice": "LogDice",
    "ll": "LogLik",
    "chisq": "ChiSq",
    "tscore": "TScore",
    "zscore": "ZScore",
    "minsens": "MinSens",
}

FREQS = ("f12", "f1", "f2", "n")


def collocations(
    results: SearchResults | LazySearchResults,
    expr: IntoExpr,
    method: str | list[str],
    window: int | tuple[int, int] = 5,
    chunk_column: Optional[str] = None,
    min_freq: int = 5,
    min_range: int = 0,
) -> pl.DataFrame:
    """
    Rank the words that occur near a search's matches by strength of attraction.

    Counts the words in a window around every match, the way `collocates`
    does, and scores each one against how often it occurs in the corpus at
    large. Words that the node word attracts occur in the windows
    more often than their own corpus frequency would predict, and come out at
    the top.

    Parameters
    ----------
    results : SearchResults or LazySearchResults
        Results of [`search`][polars_corpus.matcher.search] or
        [`search_cqp`][polars_corpus.matcher.search_cqp].
    expr : IntoExpr
        Column name or expression identifying the collocates (e.g. token or
        lemma). It must produce a single column, though that column may be a
        struct: `pl.struct("lemma", "pos")` collocates on the pair.
    method : str | list of str
        [Association metric](assoc.md) to rank the collocates by, or a list of
        them to compute together:

        - 'freq' : raw frequency in the windows, no association measure
        - 'pmi' : Pointwise Mutual Information, which favors rare words
        - 'mi3' : MI3, which pulls the ranking back towards frequent ones
        - 'logdice' : log-Dice, comparable across corpora of different sizes
        - 'll' : Log-likelihood ratio (G²)
        - 'chisq' : Pearson's chi-squared (χ²)
        - 'tscore' : t-score, which favors frequent words
        - 'zscore' : z-score
        - 'minsens' : Minimum sensitivity
    window : int or (int, int), default 5
        Words to take on each side of a match. To define an asymmetric window,
        pass a pair: `(0, 5)` collects only what follows a match and `(5, 0)` only what precedes it.
        At least one side must be > 0. Ignored when `chunk_column` is given.
    chunk_column : str, optional
        Column of BIO tags, as `concordance` takes. If provided, the window then
        runs to  the edges of the chunk holding the match rather than a fixed number of
        words.
    min_freq : int, default 5
        Minimum number of times a word must occur in context to be reported.
    min_range : int, default 0
        Minimum number of distinct files a word must occur in in
        context to be reported.

    Returns
    -------
    pl.DataFrame
        One row per collocate, sorted by the first measure asked for,
        strongest first:

        - `collocate` : the word, as `expr` reads it
        - `freqs` : a struct of the four counts the measures are computed
          from, laid out as `crosstab` lays them out -- `f12` times the word
          fell in a window, `f1` window positions filled, `f2` the word's
          corpus frequency, `n` the corpus size
        - `range` : files the word collocated in, when the results know their
          file id column
        - one column per measure, in the order asked for

    Raises
    ------
    ValueError
        If `results` is None; if `expr` does not identify a single column of the
        corpus; if `method` is not one of the measures listed above, or a list of
        them; if `window` includes no context; or if `min_range` is asked for
        over results with no file id column.

    See Also
    --------
    [search][polars_corpus.matcher.search] : Find the matches to collocate around.
    [search_cqp][polars_corpus.matcher.search_cqp] : The same, from a CQP query.
    [SearchResults.collocates][polars_corpus.search.SearchResults.collocates] :
        The window counts on their own, unranked.
    [keywords][polars_corpus.keywords.keywords] :
        The same measures over two corpora rather than a window.

    Notes
    -----
    A word in the windows of two nearby matches is counted once for each, so
    overlapping windows count their shared words twice. A word's `f12` can
    therefore come out higher than its corpus frequency `f2`.

    The matched words themselves are never collocates of the match they
    belong to, but a second occurrence of the node word nearby is.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> results = plc.search(corpus, "law")
    >>> plc.collocations(results, "token", "logdice", window=3)
    >>> # Several measures side by side, to see where they disagree:
    >>> plc.collocations(results, "lemma", ["pmi", "tscore", "logdice"])
    >>> # Colligation: the parts of speech a word keeps company with
    >>> plc.collocations(results, "pos", "ll")
    >>> # Lemma and tag together, so "spend_VERB" is its own collocate:
    >>> plc.collocations(results, pl.struct("lemma", "pos"), "ll")
    >>> # Only what follows the match, and only within its sentence:
    >>> plc.collocations(results, "lemma", "ll", window=(0, 5))
    >>> plc.collocations(results, "lemma", "ll", chunk_column="sentence_tag")
    >>> # Drop collocations that come from just a handful of texts:
    >>> plc.collocations(results, "lemma", "ll", min_freq=10, min_range=5)
    """
    methods = check_choices(method, METHODS)
    span = _window_span(window, chunk_column)
    _check_count(min_freq, "min_freq", " Use min_freq=0 to keep them all.")
    _check_count(min_range, "min_range")

    if results is None:
        raise ValueError(
            "collocations() needs the results of a search, but got None. A "
            "search that matched nothing returns None, and a node word with "
            "no matches has no windows to collect collocates from."
        )
    if not isinstance(results, (SearchResults, LazySearchResults)):
        raise ValueError(
            "results must be the matches search() or search_cqp() returned, "
            f"got {type(results).__name__}"
        )

    term = as_expr(expr)
    lf = results._frame().lazy()
    term_name = check_expr(lf, term)

    file_id_column = results._file_id_column
    if min_range and file_id_column is None:
        raise ValueError(
            "min_range counts the files a collocation occurs in, but these "
            "results have no file ids to count. Search with a "
            "file_id_column to get them."
        )

    result = results._collocate_counts(
        term, term_name, span, chunk_column, file_id_column
    )
    if min_freq:
        result = result.filter(pl.col("f12") >= min_freq)
    if min_range:
        result = result.filter(pl.col("range") >= min_range)

    measures = {
        "freq": pl.col("f12").alias("freq"),
        "pmi": pmi(*FREQS).alias("PMI"),
        "mi3": mi3(*FREQS).alias("MI3"),
        "logdice": logdice(*FREQS).alias("LogDice"),
        "ll": loglik(*FREQS).alias("LogLik"),
        "chisq": chisq(*FREQS).alias("ChiSq"),
        "tscore": tscore(*FREQS).alias("TScore"),
        "zscore": zscore(*FREQS).alias("ZScore"),
        "minsens": minsens(*FREQS).alias("MinSens"),
    }
    columns = ["collocate", pl.struct(pl.col(FREQS).cast(pl.UInt64)).alias("freqs")]
    if file_id_column is not None:
        columns.append("range")
    columns += [COLUMNS[m] for m in methods]

    result = (
        result.with_columns(*(measures[m] for m in methods))
        .select(columns)
        .sort(by=COLUMNS[methods[0]], descending=True)
    )

    return result.collect(engine="streaming")
