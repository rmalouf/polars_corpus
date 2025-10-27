from __future__ import annotations

from typing import Any, Optional

import polars as pl

from .assoc import crosstab
from .chunk import chunk_id, with_chunk_index
from .matcher import search
from .search import SearchResults
from .utils import ngrams
from .lexical import ttr, msttr


@pl.api.register_expr_namespace("corpus")
class CorpusExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def ttr(self, **kwargs: Any) -> pl.Expr:
        return ttr(self._expr, **kwargs)

    def msttr(self, **kwargs: Any) -> pl.Expr:
        return msttr(self._expr, **kwargs)

    def chunk_index(self, format: str = "IOB2") -> pl.Expr:
        if format == "IOB2":
            return (self._expr != "O" & self._expr == "B").cum_sum()
        else:
            raise NotImplementedError

    def ngrams(self, n: int) -> pl.Expr:
        return ngrams(n, self._expr)

    def chunk_id(self) -> pl.Expr:
        """Convert BIO tags to chunk IDs.

        Returns consecutive integer IDs for each chunk, with None for 'O' tags.
        Each 'B' tag starts a new chunk with an incrementing ID. 'I' tags
        continue the current chunk. 'O' tags are assigned None.

        Returns
        -------
        pl.Expr
            Expression with chunk IDs (integers) or None for outside tags.

        Examples
        --------
        >>> df = pl.DataFrame({
        ...     "bio": ["B", "I", "O", "B", "I"]
        ... })
        >>> df.with_columns(pl.col("bio").corpus.chunk_id().alias("chunk_idx"))
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
        return chunk_id(self._expr)


@pl.api.register_dataframe_namespace("corpus")
class CorpusDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def crosstab(self, x: str, y: str) -> pl.DataFrame:
        return crosstab(self._df, x, y)

    def with_chunk_index(self, chunk_col: str, **kwargs: Any) -> pl.DataFrame:
        return with_chunk_index(self._df, chunk_col, **kwargs)

    def search(self, query: str) -> Optional[SearchResults]:
        return search(self._df, query)


@pl.api.register_lazyframe_namespace("corpus")
class CorpusLazyFrame:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def crosstab(self, x: str, y: str) -> pl.LazyFrame:
        return crosstab(self._lf, x, y)

    def with_chunk_index(self, chunk_col: str, **kwargs: Any) -> pl.LazyFrame:
        return with_chunk_index(self._lf, chunk_col, **kwargs)
