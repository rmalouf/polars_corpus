from __future__ import annotations

from typing import Optional

import polars as pl

from ._typing import IntoExpr, T_Frame
from .utils import (
    DEFAULT_FILE_ID,
    as_corpus,
    as_expr,
    check_columns,
    check_expr,
    collect_like,
    proportion,
)

__all__ = [
    "frequency_list",
]


def frequency_list(
    corpus: T_Frame,
    expr: IntoExpr = "token",
    basis: float = 1_000_000,
    file_id_column: Optional[str] = DEFAULT_FILE_ID,
) -> T_Frame:
    """
    Count how often each word occurs in a corpus.

    A corpus holds one row per token, an occurrence of a word. A frequency
    list holds one row per type, a distinct form, with the number of tokens of
    it. Raw counts can only be compared within one corpus, so the count is
    also reported as a rate per `basis` words, and the number of files the
    word occurs in says whether the count comes from across the corpus or from
    a few texts.

    Parameters
    ----------
    corpus : DataFrame | LazyFrame
        Corpus to count.
    expr : IntoExpr, default "token"
        Column name or expression identifying the word/type to count
        (e.g. token or lemma).
    basis : float, default 1_000_000
        Number of words `rate` is reported per. Per million is usual for a
        large corpus; pass 10_000 for a small one.
    file_id_column : str, optional
        Column holding file ids, counted to give each word its `range`. Pass
        None for no `range` column. Defaults to "file_id", which is dropped
        rather than demanded when the corpus has no such column.

    Returns
    -------
    DataFrame | LazyFrame
        One row per type, most frequent first, ties broken by `expr` ascending:

        - the word, in a column named for whatever `expr` produces
        - `freq` : number of tokens of it
        - `rate` : `freq` as a share of the corpus, times `basis`
        - `range` : number of distinct files it occurs in, absent when there
          is no file id column

        Eager if `corpus` is a DataFrame, lazy if it is a LazyFrame.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
        missing a column `frequency_list` needs; if `expr` is not a column
        name or expression; or if `basis` is not a positive number.

    See Also
    --------
    [dispersion][polars_corpus.dispersion.dispersion] :
        How evenly a word is spread over the files, past counting them.
    [keywords][polars_corpus.keywords.keywords] :
        Which words are more frequent here than in another corpus.

    Notes
    -----
    Rows holding a null in either `expr` or `file_id_column` are dropped.

    Normalizing is `expr`'s job. Folding case is
    `pl.col("token").str.to_lowercase()`, and a token `expr` evaluates to null
    is dropped, so restricting what counts as a word is a `filter` on the
    corpus or a `when`/`then` in `expr`. The Examples show both.

    `rate` divides by the tokens that were counted, so whatever normalizing
    dropped is gone from the total too. Filtering the result afterwards does
    not rescale it: dropping the rare words leaves the common ones with the
    rates they had in the whole corpus.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.frequency_list(corpus, "token")
    >>> # Counting words rather than tokens: letters only, and case folded.
    >>> plc.frequency_list(
    ...     corpus.filter(plc.is_letters("token")),
    ...     pl.col("token").str.to_lowercase(),
    ... )
    >>> # Lemmas rather than word forms, per 10,000 words:
    >>> plc.frequency_list(corpus, "lemma", basis=10_000)
    >>> # Drop the words that are rare, or that come from a handful of texts:
    >>> plc.frequency_list(corpus, "token").filter(
    ...     pl.col("freq") >= 10, pl.col("range") >= 5
    ... )
    """
    if not isinstance(basis, (int, float)) or isinstance(basis, bool) or basis <= 0:
        raise ValueError(f"basis must be a positive number, got {basis!r}")

    term = as_expr(expr)
    lf = as_corpus(corpus)
    term_name = check_expr(lf, term)

    # An unasked-for "file_id" is a default rather than a demand: a corpus
    # without one still has a frequency list, just no range to report.
    if file_id_column == DEFAULT_FILE_ID:
        if file_id_column not in lf.collect_schema().names():
            file_id_column = None
    elif file_id_column is not None:
        check_columns(lf, [file_id_column], param="file_id_column")

    # Reading the file id column twice would duplicate it; grouping by the word
    # then counts distinct values of the word itself, which is 1 in every group.
    columns = [term]
    if file_id_column is not None and file_id_column != term_name:
        columns.append(pl.col(file_id_column))
    lf = lf.select(columns).drop_nulls()

    counts = [pl.len().alias("freq")]
    if file_id_column is not None:
        counts.append(pl.col(file_id_column).n_unique().alias("range"))

    result = (
        lf.group_by(term_name)
        .agg(counts)
        .with_columns((proportion("freq") * basis).alias("rate"))
    )

    reported = [term_name, "freq", "rate"]
    if file_id_column is not None:
        reported.append("range")
    # Ties broken by the word, so the same corpus always gives the same rows in
    # the same order.
    result = result.select(reported).sort(["freq", term_name], descending=[True, False])

    return collect_like(result, corpus)
