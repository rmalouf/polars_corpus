from __future__ import annotations

from typing import Any, Optional

import polars as pl

from .assoc import chisq, crosstab, loglik, minsens, pmi, smp
from .chunk import chunk_id, with_chunk_index
from .lexical import msttr, mtld, ttr, yules_k
from .matcher import search, search_cqp
from .search import LazySearchResults, SearchResults
from .utils import ngrams

# Which one comes back follows the frame searched; the namespace methods take
# either, so they say so.
_Results = SearchResults | LazySearchResults


@pl.api.register_expr_namespace("corpus")
class CorpusExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def ttr(self, **kwargs: Any) -> pl.Expr:
        return ttr(self._expr, **kwargs)

    def msttr(self, **kwargs: Any) -> pl.Expr:
        return msttr(self._expr, **kwargs)

    def yules_k(self, **kwargs: Any) -> pl.Expr:
        return yules_k(self._expr, **kwargs)

    def mtld(self, **kwargs: Any) -> pl.Expr:
        return mtld(self._expr, **kwargs)

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

    def loglik(self) -> pl.Expr:
        """Compute log-likelihood ratio from a freqs struct column."""
        return loglik(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def pmi(self) -> pl.Expr:
        """Compute pointwise mutual information from a freqs struct column."""
        return pmi(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def minsens(self) -> pl.Expr:
        """Compute minimum sensitivity from a freqs struct column."""
        return minsens(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def smp(self, k: float) -> pl.Expr:
        """Compute Kilgarriff's simple maths parameter from a freqs struct column."""
        return smp(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            k,
        )

    def chisq(self, yates: bool = False) -> pl.Expr:
        """Compute Pearson's chi-squared statistic from a freqs struct column."""
        return chisq(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            yates,
        )


@pl.api.register_dataframe_namespace("corpus")
class CorpusDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def crosstab(self, x: str, y: str, freqs_name: str = "freqs") -> pl.DataFrame:
        return crosstab(self._df, x, y, freqs_name)

    def with_chunk_index(self, chunk_column: str, **kwargs: Any) -> pl.DataFrame:
        return with_chunk_index(self._df, chunk_column, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search(self._df, query, **kwargs)

    def search_cqp(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search_cqp(self._df, query, **kwargs)


@pl.api.register_lazyframe_namespace("corpus")
class CorpusLazyFrame:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def crosstab(self, x: str, y: str, freqs_name: str = "freqs") -> pl.LazyFrame:
        return crosstab(self._lf, x, y, freqs_name)

    def with_chunk_index(self, chunk_column: str, **kwargs: Any) -> pl.LazyFrame:
        return with_chunk_index(self._lf, chunk_column, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search(self._lf, query, **kwargs)

    def search_cqp(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search_cqp(self._lf, query, **kwargs)
