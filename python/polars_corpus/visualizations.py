import warnings

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from matplotlib.axes import Axes

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_columns,
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
        The matplotlib axes the plot was drawn on, one row per word in
        `targets`, in the order given.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or cannot
        evaluate `expr`; or if `expr` is not a column name or expression.

    Warns
    -----
    UserWarning
        If a word in `targets` does not occur in the corpus. It still gets a
        row, drawn empty.

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

    # Filter on the term's own column rather than on `expr`: a computed term is
    # gone from the frame by the time the select is through with it.
    data = (
        lf.with_row_index()
        .select("index", term)
        .filter(pl.col(term_name).is_in(targets))
        .collect()
    )

    # Reduce before crossing into Python: `data` can hold millions of matches,
    # but at most one distinct value per target.
    found = set(data[term_name].unique().to_list())
    if missing := [target for target in targets if target not in found]:
        warnings.warn(
            f"{', '.join(repr(target) for target in missing)} "
            f"{'do' if len(missing) > 1 else 'does'} not occur in the corpus's "
            f"{term_name!r} column",
            stacklevel=2,
        )

    ax = sns.stripplot(
        x="index",
        y=term_name,
        data=data,
        # Keep the rows in the order asked for, and give a word that occurs
        # nowhere a row of its own rather than dropping it off the plot.
        order=targets,
        marker="|",
        linewidth=linewidth,
        size=size,
        jitter=jitter,
        **kwargs,
    )
    ax.set(xlabel="index", title=f"Barcode plot: {term_name}")

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
        The matplotlib axes the plot was drawn on, one row per file `target`
        occurs in.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
        missing a column `dispersion_plot` needs; if `expr` is not a column name
        or expression; or if `target` does not occur in the corpus.

    Notes
    -----
    Only the files `target` occurs in get a row, so the plot says how evenly the
    word is spread over those files, not over the corpus. Read it alongside
    `dispersion(corpus, expr, "range")`, which counts them.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.dispersion_plot(corpus, "lemma", "whale")
    """
    term = as_expr(expr)
    lf = as_corpus(corpus)

    check_columns(lf, [file_id_column], param="file_id_column")
    term_name = check_expr(lf, term)

    position = pl.int_range(1, pl.len() + 1)
    if relative:
        position = position / pl.len()
    # Filter on the term's own column rather than on `expr`: a computed term is
    # gone from the frame by the time the select is through with it.
    data = (
        lf.select(position.over(file_id_column).alias("index"), file_id_column, term)
        .filter(pl.col(term_name) == target)
        .collect()
    )

    if data.height == 0:
        raise ValueError(
            f"{target!r} does not occur in the corpus's {term_name!r} column, "
            f"so there is nothing to plot"
        )

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
    term_expr: IntoExpr,
    keyness_expr: IntoExpr,
    ax: Axes | None = None,
    top_k: int = 10,
    descending: bool = True,
    padding: int = 6,
    **kwargs,
) -> Axes:
    """
    Plot ranked keywords as a horizontal lollipop/stem chart.

    Each row of `keyword_df` becomes a stem, positioned by `keyness_expr` and
    labeled with `term_expr`, giving a quick visual read of the strongest
    keywords and how strong they are relative to each other. `keyword_df` is
    plotted in the order it is given (e.g. the output of `keywords`), so it
    should already be sorted by association strength.

    Parameters
    ----------
    keyword_df : DataFrame | LazyFrame
        Keywords to plot, e.g. the output of `keywords`.
    term_expr : IntoExpr
        Column name or expression giving the label for each stem
        (e.g. token or lemma).
    keyness_expr : IntoExpr
        Column name or expression giving the keyness/association score each
        stem is positioned at.
    ax : Axes, optional
        Axes to draw on. A new figure and axes are created if not given.
    top_k : int, default 10
        Number of rows to plot, taken from the start of `keyword_df`.
        Plots every row if `top_k` is `None` or non-positive.
    descending : bool, default True
        Plot the first row of `keyword_df` at the top rather than the bottom.
    padding : int, default 6
        Offset in points between a stem's tip and its label.
    **kwargs
        Passed through to `Axes.stem`.

    Returns
    -------
    Axes
        The matplotlib axes the plot was drawn on.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> male_keywords = plc.keywords(male_corpus, reference, "token", "ll")
    >>> plc.keyword_plot(male_keywords, "token", "LL", top_k=15)
    """
    term = as_expr(term_expr)
    keyness = as_expr(keyness_expr)
    lf = as_corpus(keyword_df)
    term_name = check_expr(lf, term)
    keyness_name = check_expr(lf, keyness)

    if top_k is not None and top_k > 0:
        lf = lf.head(top_k)

    keywords = lf.collect()

    if descending:
        indices = range(len(keywords), 0, -1)
    else:
        indices = range(len(keywords))

    if ax is None:
        _, ax = plt.subplots()

    stems = ax.stem(
        indices,
        keywords[keyness_name],
        linefmt="-",
        markerfmt="o",
        basefmt="lightgray",
        orientation="horizontal",
        **kwargs,
    )

    xs, ys = stems.markerline.get_data()
    for x, y, label in zip(xs, ys, keywords[term_name]):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(padding if x >= 0 else -padding, 0),
            textcoords="offset points",
            ha="left" if x >= 0 else "right",
            va="center_baseline",
        )

    ax.set_yticks([])
    for _, spine in ax.spines.items():
        spine.set_visible(False)

    return ax


## TODO:
## mosaic plot from crosstab
## collocation graph
