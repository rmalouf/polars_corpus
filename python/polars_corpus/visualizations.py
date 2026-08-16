import warnings

import polars as pl
import plotly.express as px
import plotly.graph_objects as go

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
    linewidth: float = 0.75,
    size: float = 20,
) -> go.Figure:
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
        Width of each tick mark, in pixels.
    size : float, default 20
        Length of each tick mark, in pixels.

    Returns
    -------
    Figure
        A plotly figure, one row per word in `targets`, in the order given.

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

    data = (
        lf.with_row_index()
        .select("index", term)
        .filter(pl.col(term_name).is_in(targets))
        .collect()
    )

    found = data[term_name].unique()
    if missing := [target for target in targets if target not in found]:
        warnings.warn(
            f"{', '.join(repr(target) for target in missing)} "
            f"{'do' if len(missing) > 1 else 'does'} not occur in the corpus's "
            f"{term_name!r} column",
            stacklevel=2,
        )

    fig = px.strip(data, "index", term_name, category_orders={term_name: targets})
    fig.update_traces(
        jitter=0,
        marker=dict(
            symbol="line-ns",
            size=size,
            line=dict(width=linewidth),
        ),
    )

    return fig


def dispersion_plot(
    corpus: T_Frame,
    expr: IntoExpr,
    target: str,
    file_id_column: str = "file_id",
    relative: bool = True,
    linewidth: float = 0.75,
    size: float = 20,
) -> go.Figure:
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
        Plot each occurrence's relative position in the file.
    linewidth : float, default 0.75
        Width of each tick mark, in pixels.
    size : float, default 20
        Length of each tick mark, in pixels.

    Returns
    -------
    Figure
        A plotly figure, one row per file `target` occurs in, in corpus order.

    Raises
    ------
    ValueError
        If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
        missing a column `dispersion_plot` needs; if `expr` is not a column name
        or expression; or if `target` does not occur in the corpus.

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
    fig = px.strip(
        data,
        "index",
        file_id_column,
        labels={"index": "relative index" if relative else "index"},
    )
    fig.update_traces(
        jitter=0,
        marker=dict(
            symbol="line-ns",
            size=size,
            line=dict(width=linewidth),
        ),
    )

    return fig


def keyword_plot(
    keyword_df: T_Frame,
    term_expr: IntoExpr,
    keyness_expr: IntoExpr,
    top_k: int | None = 10,
    descending: bool = True,
    reverse: bool = False,
) -> go.Figure:
    """
    Plot ranked keywords as a horizontal lollipop/stem chart.

    Each row of `keyword_df` becomes a stem, positioned by `keyness_expr` and
    labeled with `term_expr`, giving a quick visual read of the top
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
    top_k : int | None, default 10
        Number of rows to plot, taken from the start of `keyword_df`.
        Plots every row if `top_k` is `None` or non-positive.
    descending : bool, default True
        Plot the first row of `keyword_df` at the top rather than the bottom.
    reverse : bool, default False
        Mirror the chart, so the stems run leftwards from the axis and the
        labels sit to the left of their markers. Useful for putting two
        keyword plots back to back.

    Returns
    -------
    Figure
        A plotly figure.

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

    keywords = lf.select(term, keyness).collect()

    if keywords.height == 0:
        raise ValueError("the keyword_df is empty, so there is nothing to plot")

    if reverse:
        text_position = "middle left"
    else:
        text_position = "middle right"

    bar_x = []
    bar_y = []
    y = list(range(len(keywords)))
    for xi, yi in zip(keywords[keyness_name], y):
        bar_x += [0, xi, None]
        bar_y += [yi, yi, None]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=bar_x,
            y=bar_y,
            mode="lines",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=keywords[keyness_name],
            y=y,
            mode="markers+text",
            text=keywords[term_name],
            textposition=text_position,
            marker=dict(size=10),
            showlegend=False,
        )
    )

    fig.update_yaxes(
        showticklabels=False,
        ticks="",
        showgrid=False,
        zeroline=False,
    )

    if descending:
        fig.update_yaxes(autorange="reversed")

    fig.update_xaxes(
        rangemode="tozero",
        # showgrid=False,
        # zeroline=False,
    )

    if reverse:
        fig.update_xaxes(autorange="reversed")

    fig.update_traces(cliponaxis=False)

    return fig


def text_plot(points, labels, show_labels: bool = True) -> go.Figure:
    """
    Plot labeled points on a two-dimensional map.

    Draws each row of `points` with its label, for reading a projection of
    embedding vectors (e.g. from UMAP) down to two dimensions. The axes carry
    no scale, since the coordinates only mean something relative to each other.

    Parameters
    ----------
    points : array-like of shape (n, 2)
        Coordinates to plot, one row per label.
    labels : sequence of str
        Label for each row of `points`, e.g. the token or the concordance line the
        vector was built from.
    show_labels : bool, default True
        Draw the labels. When False, the points are drawn as a plain scatter
        instead.

    Returns
    -------
    Figure
        A plotly figure.

    Examples
    --------
    >>> import umap
    >>> xy = umap.UMAP().fit_transform(df["vector"].to_numpy())
    >>> plc.text_plot(xy, df["token"])
    >>> plc.text_plot(xy, df["token"], show_labels=False)
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            text=labels,
            mode="markers+text" if show_labels else "markers",
            textposition="top right",
            showlegend=False,
            cliponaxis=False,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


## TODO:
## mosaic plot from crosstab
## collocation graph
