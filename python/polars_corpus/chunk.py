"""Chunk and span indexing utilities for BIO-tagged data."""

from __future__ import annotations

import polars as pl

from ._typing import T_Frame

__all__ = ["chunk_id", "with_chunk_index"]


def chunk_id(expr: pl.Expr) -> pl.Expr:
    """Convert BIO tags to chunk IDs.

    Returns consecutive integer IDs for each chunk, with None for 'O' tags.
    Each 'B' tag starts a new chunk with an incrementing ID. 'I' tags
    continue the current chunk. 'O' tags are assigned None.

    Parameters
    ----------
    expr
        Expression containing BIO tags.

    Returns
    -------
    pl.Expr
        Expression with chunk IDs (integers) or None for outside tags.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_corpus import chunk_id
    >>> df = pl.DataFrame({
    ...     "bio": ["B", "I", "O", "B", "I"]
    ... })
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
    """Add a column with chunk IDs based on BIO tags.

    Parameters
    ----------
    df
        DataFrame or LazyFrame with BIO-tagged column.
    chunk_column
        Name of column containing BIO tags.
    name
        Name for the new chunk ID column (default: "chunk_idx").

    Returns
    -------
    T_Frame
        DataFrame or LazyFrame with added chunk ID column.

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
