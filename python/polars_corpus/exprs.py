from __future__ import annotations

from typing import Any, Optional

import polars as pl

from ._typing import IntoExpr
from .assoc import (
    bic,
    chisq,
    crosstab,
    logdice,
    loglik,
    logratio,
    mi3,
    minsens,
    oddsratio,
    pctdiff,
    pmi,
    smp,
    tscore,
    zscore,
)
from .chunk import chunk_id, with_chunk_index
from .frequency import frequency_list
from .lexical import (
    count_hapaxes,
    frequency_spectrum,
    msttr,
    mtld,
    ttr,
    vocabulary_growth,
    yules_k,
)
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

    def vocabulary_growth(self) -> pl.Expr:
        return vocabulary_growth(self._expr)

    def frequency_spectrum(self, sort: bool = False) -> pl.Expr:
        return frequency_spectrum(self._expr, sort=sort)

    def count_hapaxes(self) -> pl.Expr:
        return count_hapaxes(self._expr)

    def ngrams(self, n: int) -> pl.Expr:
        return ngrams(n, self._expr)

    def chunk_id(self) -> pl.Expr:
        """
        Number the chunks this column of BIO tags marks out.

        Returns
        -------
        pl.Expr
            Expression giving each token the number of "B" tags at or before
            it, so 1 for the first chunk and 2 for the second, and null where
            its tag is "O".

        See Also
        --------
        polars_corpus.chunk_id : Function form, with the details.

        Examples
        --------
        >>> df = pl.DataFrame({"bio": ["B", "I", "O", "B", "I"]})
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

    def mi3(self) -> pl.Expr:
        """Compute the MI3 association measure from a freqs struct column."""
        return mi3(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def logdice(self) -> pl.Expr:
        """Compute log-Dice from a freqs struct column."""
        return logdice(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def tscore(self) -> pl.Expr:
        """Compute the t-score from a freqs struct column."""
        return tscore(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def zscore(self) -> pl.Expr:
        """Compute the z-score from a freqs struct column."""
        return zscore(
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

    def bic(self) -> pl.Expr:
        """Compute the Bayes factor BIC from a freqs struct column."""
        return bic(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def logratio(self, discount: float = 0.5) -> pl.Expr:
        """Compute Hardie's log ratio from a freqs struct column."""
        return logratio(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )

    def pctdiff(self, discount: float = 0.5) -> pl.Expr:
        """Compute %DIFF from a freqs struct column."""
        return pctdiff(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )

    def oddsratio(self, discount: float = 0.5) -> pl.Expr:
        """Compute the odds ratio from a freqs struct column."""
        return oddsratio(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )


@pl.api.register_dataframe_namespace("corpus")
class CorpusDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def crosstab(self, x: str, y: str, freqs_name: str = "freqs") -> pl.DataFrame:
        return crosstab(self._df, x, y, freqs_name)

    def frequency_list(self, expr: IntoExpr = "token", **kwargs: Any) -> pl.DataFrame:
        return frequency_list(self._df, expr, **kwargs)

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

    def frequency_list(self, expr: IntoExpr = "token", **kwargs: Any) -> pl.LazyFrame:
        return frequency_list(self._lf, expr, **kwargs)

    def with_chunk_index(self, chunk_column: str, **kwargs: Any) -> pl.LazyFrame:
        return with_chunk_index(self._lf, chunk_column, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search(self._lf, query, **kwargs)

    def search_cqp(self, query: str, **kwargs: Any) -> Optional[_Results]:
        return search_cqp(self._lf, query, **kwargs)
