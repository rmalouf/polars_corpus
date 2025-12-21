import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from ._typing import IntoExpr, T_Frame

__all__ = ["distribution_plot"]


def distribution_plot(
    df: T_Frame, expr: IntoExpr, words: str | list[str], **kwargs
) -> Axes:
    if isinstance(df, pl.DataFrame):
        df = df.lazy()
    if isinstance(expr, str):
        expr = pl.col(expr)
    if isinstance(words, str):
        words = [words]

    data = df.with_row_index().select("index", expr).filter(expr.is_in(words)).collect()

    # TODO: fix the y column name
    return sns.stripplot(x="index", y="token", data=data, **kwargs)


## mosaic plot from crosstab
## collocation graph
