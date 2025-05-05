from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import polars as pl
from polars.plugins import register_plugin_function

from .assoc import crosstab, assoc
from .concordance import concordance

LIB = Path(__file__).parent


if TYPE_CHECKING:
    from polars.type_aliases import IntoExprColumn


all = ["whichlang"]


@pl.api.register_expr_namespace("text")
class TextExpr:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def chunk_index(self, format: str = "IOB2") -> pl.Expr:
        if format == "IOB2":
            return (self._expr != "O" & self._expr == "B").cum_sum()
        else:
            raise NotImplementedError

    def ngrams(self, n: int) -> pl.Expr:
        return pl.concat_list(self._expr.shift(-i) for i in range(0, n))

    def whichlang(self) -> pl.Expr:
        return whichlang(self._expr)


@pl.api.register_dataframe_namespace("text")
class TextDataFrame:
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

    def concordance(
        self,
        by: pl.Expr,
        context: int,
        left_context: Optional[int],
        right_context: Optional[int],
    ) -> pl.DataFrame:
        return concordance(self._df, by, context, left_context, right_context)


@pl.api.register_lazyframe_namespace("text")
class TextLazyFrame:
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

    def concordance(
        self,
        by: pl.Expr,
        context: int,
        left_context: Optional[int],
        right_context: Optional[int],
    ) -> pl.LazyFrame:
        return concordance(self._lf, by, context, left_context, right_context)


def whichlang(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="whichlang",
        is_elementwise=True,
    )
