import polars as pl

from .collocations import tabulate


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


@pl.api.register_dataframe_namespace("text")
class TextDataFrame:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def tabulate(self, x: str, y: str) -> pl.DataFrame:
        return tabulate(self._df, x, y)


@pl.api.register_lazyframe_namespace("text")
class TextLazyFrame:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def tabulate(self, x: str, y: str) -> pl.LazyFrame:
        return tabulate(self._lf, x, y)
