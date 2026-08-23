"""Chunk and span indexing utilities for BIO-tagged data."""

from __future__ import annotations

import polars as pl

from ._typing import T_Frame

__all__ = ["chunk_id", "with_chunk_index"]


def chunk_id(expr: pl.Expr) -> pl.Expr:
    """
    Number the chunks that a column of BIO tags marks out.

    BIO tagging marks a run of tokens with "B" on its first token and "I" on
    the rest. Tokens outside any chunk are tagged "O". A sentence, a noun
    phrase, or a search match can all be marked this way.

    This counts the "B" tags. Each one starts a new chunk and increases the
    count by one, an "I" keeps the count of the chunk it continues, and an "O"
    comes out null. Every token of a chunk therefore gets the same number, and
    no two chunks get the same one, which is what `group_by` needs to collect
    a chunk's tokens together.

    Parameters
    ----------
    expr : pl.Expr
        Expression holding the BIO tags.

    Returns
    -------
    pl.Expr
        Expression giving each token the number of "B" tags at or before it,
        and null where its tag is "O". The first chunk is 1, the second 2, and
        so on. A token tagged "I" before any "B" has appeared gets 0.

    Notes
    -----
    The count runs the length of the frame, so the numbers are unique across
    the whole corpus and two files' chunks never collide.

    The drawback is that a chunk left open at the end of a file runs into the
    next one. If the following file starts with "I" rather than "B", its first
    tokens join the chunk before them. Add `.over()` on the file id column to
    count within each file instead, which stops that but makes the numbers
    repeat from file to file.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import chunk_id
    >>> df = pl.DataFrame({"bio": ["B", "I", "O", "B", "I"]})
    >>> df.with_columns(chunk_id(pl.col("bio")).alias("chunk_idx"))
    shape: (5, 2)
    ┌─────┬───────────┐
    │ bio ┆ chunk_idx │
    │ --- ┆ ---       │
    │ str ┆ i64       │
    ╞═════╪═══════════╡
    │ B   ┆ 1         │
    │ I   ┆ 1         │
    │ O   ┆ null      │
    │ B   ┆ 2         │
    │ I   ┆ 2         │
    └─────┴───────────┘
    """
    is_b = expr.eq("B")
    chunk_idx = is_b.cum_sum()
    return pl.when(expr.eq("O")).then(None).otherwise(chunk_idx)


def with_chunk_index(
    df: T_Frame, chunk_column: str, name: str = "chunk_idx"
) -> T_Frame:
    """
    Add a column numbering the chunks that a column of BIO tags marks out.

    Parameters
    ----------
    df : DataFrame | LazyFrame
        Corpus holding the tags.
    chunk_column : str
        Column holding the BIO tags, e.g. a sentence tag, or the column
        `SearchResults.with_spans_as_chunks` adds.
    name : str, default "chunk_idx"
        Name for the added column.

    Returns
    -------
    T_Frame
        The corpus, one row per token as before, with a column numbering the
        chunks as `chunk_id` describes: 1 for the first chunk, 2 for the
        second, null for a token tagged "O". Eager if `df` is a DataFrame,
        lazy if it is a LazyFrame.

    See Also
    --------
    chunk_id : The expression this adds, to put in a query of your own.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import with_chunk_index
    >>> df = pl.DataFrame({"bio": ["B", "I", "O", "B", "I"]})
    >>> with_chunk_index(df, "bio")
    shape: (5, 2)
    ┌─────┬───────────┐
    │ bio ┆ chunk_idx │
    │ --- ┆ ---       │
    │ str ┆ i64       │
    ╞═════╪═══════════╡
    │ B   ┆ 1         │
    │ I   ┆ 1         │
    │ O   ┆ null      │
    │ B   ┆ 2         │
    │ I   ┆ 2         │
    └─────┴───────────┘
    """
    return df.with_columns(chunk_id(pl.col(chunk_column)).alias(name))
