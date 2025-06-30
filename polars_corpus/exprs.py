from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from .assoc import assoc, crosstab
from .matcher import search
from .search import SearchResults

LIB = Path(__file__).parent


if TYPE_CHECKING:
    from polars.type_aliases import IntoExprColumn

@pl.api.register_expr_namespace("corpus")
class CorpusExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def chunk_index(self, format: str = "IOB2") -> pl.Expr:
        if format == "IOB2":
            return (self._expr != "O" & self._expr == "B").cum_sum()
        else:
            raise NotImplementedError

    def ngrams(self, n: int) -> pl.Expr:
        return pl.concat_list(self._expr.shift(-i) for i in range(0, n))

    def kwic_concordance(
        self, search_results: SearchResults, window_size
    ) -> pl.DataFrame:
        return search_results.kwic_concordance(self._expr, window_size)

    def matches(self, search_results: SearchResults) -> pl.DataFrame:
        return search_results.matches(self._expr)


@pl.api.register_dataframe_namespace("corpus")
class CorpusDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def crosstab(self, x: str, y: str) -> pl.DataFrame:
        return crosstab(self._df, x, y)

    def mi(self, x: str, y: str, **kwargs: Any) -> pl.DataFrame:
        return assoc(self._df, x, y, "mi", **kwargs)

    def min_sens(self, x: str, y: str, **kwargs: Any) -> pl.DataFrame:
        return assoc(self._df, x, y, "min_sens", **kwargs)

    def assoc(self, x: str, y: str, method: str, **kwargs: Any) -> pl.DataFrame:
        return assoc(self._df, x, y, method, **kwargs)

    def with_span_index(self, span_col: str, **kwargs: Any) -> pl.DataFrame:
        return with_span_index(self, span_col, **kwargs)

    def search(self, query: str) -> SearchResults:
        return search(self._df, query)

    def kwic_concordance(
        self, search_results: SearchResults, expr: pl.Expr, window_size,
    ) -> pl.DataFrame:
        return search_results.kwic_concordance(expr, window_size)

    def matches(
        self, search_results: SearchResults, expr: pl.Expr, **kwargs
    ) -> pl.Expr:
        return search_results.matches(expr)


@pl.api.register_lazyframe_namespace("corpus")
class CorpusLazyFrame:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def crosstab(self, x: str, y: str) -> pl.LazyFrame:
        return crosstab(self._lf, x, y)

    def mi(self, x: str, y: str, **kwargs: Any) -> pl.LazyFrame:
        return assoc(self._lf, x, y, "mi", **kwargs)

    def min_sens(self, x: str, y: str, **kwargs: Any) -> pl.LazyFrame:
        return assoc(self._lf, x, y, "min_sens", **kwargs)

    def assoc(self, x: str, y: str, method: str, **kwargs: Any) -> pl.LazyFrame:
        return assoc(self._lf, x, y, method, **kwargs)
