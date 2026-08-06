import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_expr,
)

__all__ = ["barcode_plot", "dispersion_plot", "keyword_plot"]


def barcode_plot(
    corpus: T_Frame,
    expr: IntoExpr,
    targets: str | list[str],
    linewidth: float = 0.75,
    size: float = 20,
    jitter: float | bool = 0,
    **kwargs,
) -> Axes:
    """
    Plot the position of one or more words across a corpus as a barcode.

    Each occurrence of a target word is drawn as a short vertical tick at its
    position in the corpus, one row per word, giving a quick visual read of
    how evenly a word is spread out.

    Parameters
    ----------
    corpus : DataFrame | LazyFrame
        Corpus to plot.
    expr : IntoExpr
        Column name or expression identifying the word/type to match
        against `targets` (e.g. token or lemma).
    targets : str | list[str]
        Word or words to plot, one row per word.
    linewidth : float, default 0.75
        Width of each tick mark.
    size : float, default 20
        Height of each tick mark.
    jitter : float | bool, default 0
        Vertical jitter to apply to ticks, as in `seaborn.stripplot`.
    **kwargs
        Passed through to `seaborn.stripplot`.

    Returns
    -------
    Axes
        The matplotlib axes the plot was drawn on.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.barcode_plot(corpus, "lemma", ["cat", "dog"])
    """
    term = as_expr(expr)
    lf = as_corpus(corpus)
    term_name = check_expr(lf, term)

    if isinstance(targets, str):
        targets = [targets]

    data = (
        lf.with_row_index().select("index", expr).filter(term.is_in(targets)).collect()
    )

    ax = sns.stripplot(
        x="index",
        y=term_name,
        data=data,
        marker="|",
        linewidth=linewidth,
        size=size,
        jitter=jitter,
        **kwargs,
    )

    return ax


def dispersion_plot(
    corpus: T_Frame,
    expr: IntoExpr,
    target: str,
    file_id_column: str = "file_id",
    relative: bool = True,
    linewidth: float = 0.75,
    size: float = 20,
    jitter: float | bool = 0,
    **kwargs,
) -> Axes:
    """
    Plot the position of a word across the files of a corpus.

    Each occurrence of `target` is drawn as a short vertical tick at its
    position within its file, one row per file, giving a quick visual read
    of how evenly the word is spread across the corpus.

    Parameters
    ----------
    corpus : DataFrame | LazyFrame
        Corpus to plot.
    expr : IntoExpr
        Column name or expression identifying the word/type to match
        against `target` (e.g. token or lemma).
    target : str
        Word to plot.
    file_id_column : str, default "file_id"
        Column holding file ids, defining the rows of the plot.
    relative : bool, default True
        Plot each occurrence's position as a fraction of its file's length
        rather than a raw index, so files of different lengths line up.
    linewidth : float, default 0.75
        Width of each tick mark.
    size : float, default 20
        Height of each tick mark.
    jitter : float | bool, default 0
        Vertical jitter to apply to ticks, as in `seaborn.stripplot`.
    **kwargs
        Passed through to `seaborn.stripplot`.

    Returns
    -------
    Axes
        The matplotlib axes the plot was drawn on.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.dispersion_plot(corpus, "lemma", "whale")
    """
    term = as_expr(expr)
    lf = as_corpus(corpus)

    if relative:
        data = lf.with_columns(
            (pl.int_range(1, pl.len() + 1) / pl.len())
            .over(file_id_column)
            .alias("index")
        )
    else:
        data = lf.with_columns(
            pl.int_range(1, pl.len() + 1).over(file_id_column).alias("index")
        )
    data = data.select("index", file_id_column, expr).filter(term == target).collect()

    ax = sns.stripplot(
        x="index",
        y=file_id_column,
        data=data,
        marker="|",
        linewidth=linewidth,
        size=size,
        jitter=jitter,
        **kwargs,
    )
    ax.set(
        xlabel="relative index" if relative else "index",
        title=f"Dispersion plot: {target}",
    )

    return ax


def keyword_plot(
    keyword_df: T_Frame,
    expr: IntoExpr,
    keyness: IntoExpr,
    top_k: int = 10,
    bottom_k: int = 10,
    **kwargs,
) -> Axes:

    term = as_expr(expr)
    lf = as_corpus(keyword_df)
    term_name = check_expr(lf, term)
    keyness_name = check_expr(lf, keyness)

    top_items = lf.head(top_k).collect()
    ax = sns.barplot(x=keyness_name, y=term_name, data=top_items, orient="h", **kwargs)
    ax.bar_label(ax.containers[0], labels=top_items[term_name], padding=4)

    bottom_items = lf.tail(bottom_k).collect()
    sns.barplot(
        x=keyness_name, y=term_name, data=bottom_items, orient="h", ax=ax, **kwargs
    )
    ax.bar_label(ax.containers[1], labels=bottom_items[term_name], padding=4)

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.margins(x=0.2)

    return ax


## TODO:
## mosaic plot from crosstab
## collocation graph
