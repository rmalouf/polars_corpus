import warnings

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
import mplcursors

from ._typing import IntoExpr, T_Frame
from .utils import (
    as_corpus,
    as_expr,
    check_columns,
    check_expr,
)

__all__ = ["barcode_plot", "dispersion_plot", "keyword_plot", "text_plot"]


def barcode_plot(
    corpus: T_Frame,
    expr: IntoExpr,
    targets: str | list[str],
    ax: Axes | None = None,
    linewidth: float = 0.75,
    size: float = 200,
    margin: float | None = None,
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
    ax : Axes, optional
        Axes to draw on. A new figure and axes are created if not given.
    linewidth : float, default 0.75
        Width of each tick mark.
    size : float, default 200
        Size of each tick mark, in points squared.
    margin : float, optional
        Extra vertical padding above the first row and below the last, in
        row heights. Rows are half a row apart from the edges without it.
    **kwargs
        Passed through to `Axes.scatter`.

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
        .with_columns(
            pl.col(term_name).replace_strict(
                {target: row for row, target in enumerate(targets)}
            )
        )
        .collect()
    )

    found = data[term_name].unique()
    if missing := [target for row, target in enumerate(targets) if row not in found]:
        warnings.warn(
            f"{', '.join(repr(target) for target in missing)} "
            f"{'do' if len(missing) > 1 else 'does'} not occur in the corpus's "
            f"{term_name!r} column",
            stacklevel=2,
        )

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        data["index"],
        data[term_name],
        marker="|",
        linewidth=linewidth,
        sizes=[size],
        **kwargs,
    )

    ax.set(xlabel="index")

    # Set the rows rather than autoscaling to them: a target that occurs nowhere
    # has no points to scale to, but still gets a row of its own.
    pad = 0.5 + (margin or 0)
    ax.set_yticks(range(len(targets)), targets)
    ax.set_ylim(len(targets) - 1 + pad, -pad)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def dispersion_plot(
    corpus: T_Frame,
    expr: IntoExpr,
    target: str,
    file_id_column: str = "file_id",
    relative: bool = True,
    ax: Axes | None = None,
    linewidth: float = 0.75,
    size: float = 200,
    margin: float | None = None,
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
    ax : Axes, optional
        Axes to draw on. A new figure and axes are created if not given.
    linewidth : float, default 0.75
        Width of each tick mark.
    size : float, default 200
        Size of each tick mark, in points squared.
    margin : float, optional
        Extra vertical padding above the first row and below the last, in
        row heights. Rows are half a row apart from the edges without it.
    **kwargs
        Passed through to `Axes.scatter`.

    Returns
    -------
    Axes
        The matplotlib axes the plot was drawn on, one row per file `target`
        occurs in, in corpus order.

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

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        data["index"],
        data[file_id_column],
        marker="|",
        linewidth=linewidth,
        sizes=[size],
        **kwargs,
    )

    ax.set(xlabel="relative index" if relative else "index")

    # Rows read top to bottom, in corpus order, as in `barcode_plot`.
    pad = 0.5 + (margin or 0)
    ax.set_ylim(data[file_id_column].n_unique() - 1 + pad, -pad)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def keyword_plot(
    keyword_df: T_Frame,
    term_expr: IntoExpr,
    keyness_expr: IntoExpr,
    ax: Axes | None = None,
    top_k: int | None = 10,
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
    top_k : int | None, default 10
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

    Raises
    ------
    ValueError
        If `keyword_df` is not a Polars DataFrame or LazyFrame, is empty, or
        cannot evaluate `term_expr` or `keyness_expr`; or if either of those is
        not a column name or expression.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> male_keywords = plc.keywords(male_corpus, reference, "token", "ll")
    >>> plc.keyword_plot(male_keywords, "token", "LogLik", top_k=15)
    """
    term = as_expr(term_expr, param="term_expr")
    keyness = as_expr(keyness_expr, param="keyness_expr")
    lf = as_corpus(keyword_df, name="keyword_df")
    term_name = check_expr(lf, term, name="keyword_df", param="term_expr")
    keyness_name = check_expr(lf, keyness, name="keyword_df", param="keyness_expr")

    if top_k is not None and top_k > 0:
        lf = lf.head(top_k)

    # Select the two expressions rather than collecting the frame whole: a
    # computed term or score exists only once evaluated, and a keyword table
    # carries columns (frequency structs, document counts) the plot never reads.
    keywords = lf.select(term, keyness).collect()

    if keywords.height == 0:
        raise ValueError("the keyword_df is empty, so there is nothing to plot")

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
    for spine in ax.spines.values():
        spine.set_visible(False)

    return ax



def text_plot(
        xy, labels, ax:Axes|None=None, adjust:bool=True, show_labels: bool = True,
        hover: bool= False
) -> Axes:

    try:
        from adjustText import adjust_text
    except ImportError:
        raise ImportError()

    if ax is None:
        _, ax = plt.subplots()

    if show_labels:
        size = 0
    else:
        size = 5

    scatter = ax.scatter(xy[:,0], xy[:,1], s=size)

    if hover:
        raise NotImplementedError("hover doesn't work yet")
        # cursor = mplcursors.cursor(scatter, hover=True)
        # @cursor.connect("add")
        # def on_add(sel):
        #     sel.annotation.set_text(labels[sel.index])

    if show_labels:
        texts = [ ]
        for point, label in zip(xy, labels):
            texts.append(ax.text(point[0], point[1], label, ha='center', va='center'))

        if adjust:
            adjust_text(texts, ax=ax, time_lim=2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)

    return ax




## TODO:
## mosaic plot from crosstab
## collocation graph
