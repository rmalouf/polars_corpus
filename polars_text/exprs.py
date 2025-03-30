import polars as pl

all = ["Text"]


@pl.api.register_expr_namespace("text")
class Text:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def chunk_index(self, format: str = "IOB2") -> pl.Expr:
        if format == "IOB2":
            return (self._expr != "O" & self._expr == "B").cum_sum()
        else:
            raise NotImplementedError

    def ngrams(self, n: int) -> pl.Expr:
        return pl.concat_list(self._expr.shift(-i) for i in range(0, n))
