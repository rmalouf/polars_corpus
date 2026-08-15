import warnings

import polars as pl
import pytest

# Plotting is the "examples" extra, not a core dependency. matplotlib comes in
# because `visualizations` still draws `text_plot` with it.
pytest.importorskip("plotly")
pytest.importorskip("matplotlib")

# Imported after the skip checks, hence not at the top.
from polars_corpus import barcode_plot, dispersion_plot, keyword_plot  # noqa: E402

# Two files of four tokens each. "cat" is in both, "dog" only in f1, and "eel"
# occurs nowhere.
CORPUS = pl.DataFrame(
    {
        "token": ["cat", "dog", "the", "dog", "the", "cat", "the", "the"],
        "file_id": ["f1"] * 4 + ["f2"] * 4,
    }
)

# A keyword table as `keywords()` returns one: ranked, strongest first.
KEYWORDS = pl.DataFrame({"token": ["cat", "dog", "eel"], "LL": [9.0, 4.0, 1.0]})


def y_labels(fig) -> list[str]:
    # A strip plot's rows, in the order the plot puts them. Plotly draws the
    # first category at the bottom, so an explicit order comes back reversed;
    # without one the rows are whatever the data hit, in corpus order.
    categories = fig.layout.yaxis.categoryarray
    if categories is not None:
        return list(reversed(categories))
    return list(dict.fromkeys(y for trace in fig.data for y in trace.y))


def x_values(fig) -> list[float]:
    return [x for trace in fig.data for x in trace.x]


def markers(fig):
    # The keyword plot's labeled markers; the other trace is the stems.
    (trace,) = [trace for trace in fig.data if trace.mode == "markers+text"]
    return trace


def test_barcode_plot_rows_follow_targets() -> None:
    # Row order is the caller's, not whichever word the corpus happens to hit first.
    fig = barcode_plot(CORPUS, "token", ["the", "cat"])
    assert y_labels(fig) == ["the", "cat"]


def test_barcode_plot_accepts_a_single_target() -> None:
    fig = barcode_plot(CORPUS, "token", "cat")
    assert y_labels(fig) == ["cat"]


def test_barcode_plot_absent_target_warns_and_keeps_its_row() -> None:
    # An absent word is worth seeing as an empty row: that is the finding.
    with pytest.warns(UserWarning, match="'eel' does not occur"):
        fig = barcode_plot(CORPUS, "token", ["cat", "eel"])
    assert y_labels(fig) == ["cat", "eel"]


def test_barcode_plot_names_every_absent_target() -> None:
    with pytest.warns(UserWarning, match="'eel', 'ox' do not occur"):
        barcode_plot(CORPUS, "token", ["eel", "ox"])


def test_barcode_plot_does_not_warn_for_present_targets() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        barcode_plot(CORPUS, "token", ["cat", "dog"])


def test_barcode_plot_computed_term() -> None:
    # The term is gone from the frame once selected, so the filter has to name
    # the column it produced rather than re-evaluate the expression.
    corpus = pl.DataFrame({"token": ["Cat", "CAT", "dog"], "file_id": ["f1"] * 3})
    fig = barcode_plot(corpus, pl.col("token").str.to_lowercase().alias("norm"), "cat")
    assert y_labels(fig) == ["cat"]
    # Both spellings are drawn, at their positions in the corpus.
    assert x_values(fig) == [0, 1]


def test_barcode_plot_rows_key_off_the_term_column() -> None:
    # The row order is set on whichever column the term produced, so it holds
    # for a corpus annotated under another name.
    fig = barcode_plot(CORPUS.rename({"token": "word"}), "word", ["the", "cat"])
    assert y_labels(fig) == ["the", "cat"]


def test_dispersion_plot_rows_are_the_files_hit() -> None:
    fig = dispersion_plot(CORPUS, "token", "dog")
    # "dog" is in f1 only, so f2 gets no row.
    assert y_labels(fig) == ["f1"]
    assert y_labels(dispersion_plot(CORPUS, "token", "cat")) == ["f1", "f2"]


@pytest.mark.parametrize(
    "relative,expected", [(True, [0.25, 0.5]), (False, [1.0, 2.0])]
)
def test_dispersion_plot_positions(relative: bool, expected: list[float]) -> None:
    # "cat" is the 1st token of f1 and the 2nd of f2, each four tokens long, so
    # relative positions are counted within the file rather than across the corpus.
    fig = dispersion_plot(CORPUS, "token", "cat", relative=relative)
    assert sorted(x_values(fig)) == pytest.approx(expected)


def test_dispersion_plot_relative_index_is_a_fraction() -> None:
    fig = dispersion_plot(CORPUS, "token", "cat", relative=True)
    assert all(0 < x <= 1 for x in x_values(fig))
    assert fig.layout.xaxis.title.text == "relative index"


def test_dispersion_plot_absolute_index_is_labeled() -> None:
    fig = dispersion_plot(CORPUS, "token", "cat", relative=False)
    assert fig.layout.xaxis.title.text == "index"


def test_dispersion_plot_absent_target_raises() -> None:
    # Left to itself the plot would come out blank, with a numeric y axis.
    with pytest.raises(ValueError, match="'eel' does not occur"):
        dispersion_plot(CORPUS, "token", "eel")


@pytest.mark.parametrize(
    "term",
    [
        pl.col("token").str.to_lowercase().alias("norm"),  # aliased
        pl.col("token").str.to_lowercase(),  # keeps root name "token"
    ],
)
def test_dispersion_plot_computed_term(term: pl.Expr) -> None:
    # The term is gone from the frame once selected, so the filter has to name
    # the column it produced rather than re-evaluate the expression -- which an
    # aliased term cannot survive, its root column having been left behind.
    corpus = pl.DataFrame(
        {"token": ["Cat", "CAT", "dog"], "file_id": ["f1", "f2", "f1"]}
    )
    fig = dispersion_plot(corpus, term, "cat")
    assert y_labels(fig) == ["f1", "f2"]


def test_dispersion_plot_file_id_column() -> None:
    fig = dispersion_plot(
        CORPUS.rename({"file_id": "text_id"}), "token", "cat", file_id_column="text_id"
    )
    assert y_labels(fig) == ["f1", "f2"]


@pytest.mark.parametrize("plot", [barcode_plot, dispersion_plot])
def test_plot_missing_term_column(plot) -> None:
    with pytest.raises(ValueError, match="the corpus has no column 'lemma'"):
        plot(CORPUS, "lemma", "cat")


def test_dispersion_plot_missing_file_id_column() -> None:
    with pytest.raises(ValueError, match="Use file_id_column= to point at"):
        dispersion_plot(CORPUS.drop("file_id"), "token", "cat")


@pytest.mark.parametrize("plot", [barcode_plot, dispersion_plot])
def test_plot_invalid_corpus(plot) -> None:
    with pytest.raises(ValueError, match="the corpus must be a polars"):
        plot("corpus", "token", "cat")
    with pytest.raises(ValueError, match="the corpus is empty"):
        plot(CORPUS.clear(), "token", "cat")


@pytest.mark.parametrize("plot", [barcode_plot, dispersion_plot])
def test_plot_lazy_corpus(plot) -> None:
    assert y_labels(plot(CORPUS.lazy(), "token", "cat")) == y_labels(
        plot(CORPUS, "token", "cat")
    )


def test_keyword_plot_labels_every_stem() -> None:
    fig = keyword_plot(KEYWORDS, "token", "LL")
    assert list(markers(fig).text) == ["cat", "dog", "eel"]


def test_keyword_plot_top_k() -> None:
    fig = keyword_plot(KEYWORDS, "token", "LL", top_k=2)
    assert list(markers(fig).text) == ["cat", "dog"]


def test_keyword_plot_computed_term() -> None:
    # A computed term exists only in the select, so the labels have to be read
    # from there rather than from the frame the caller handed over.
    fig = keyword_plot(KEYWORDS, pl.col("token").str.to_uppercase(), "LL")
    assert list(markers(fig).text) == ["CAT", "DOG", "EEL"]


def test_keyword_plot_computed_keyness() -> None:
    fig = keyword_plot(KEYWORDS, "token", (pl.col("LL") * 2).alias("scaled"))
    assert list(markers(fig).x) == [18.0, 8.0, 2.0]


def test_keyword_plot_stems_reach_the_markers() -> None:
    # Each stem is a null-separated segment from the axis out to its marker.
    fig = keyword_plot(KEYWORDS, "token", "LL")
    (stems,) = [trace for trace in fig.data if trace.mode == "lines"]
    assert list(stems.x) == [0, 9.0, None, 0, 4.0, None, 0, 1.0, None]
    assert list(stems.y) == [0, 0, None, 1, 1, None, 2, 2, None]


@pytest.mark.parametrize(
    "kwargs,axis,expected",
    [
        ({}, "yaxis", "reversed"),  # descending: the strongest keyword on top
        ({"descending": False}, "yaxis", None),
        ({}, "xaxis", None),
        ({"reverse": True}, "xaxis", "reversed"),  # mirrored, to pair two plots
    ],
)
def test_keyword_plot_axis_direction(kwargs, axis: str, expected) -> None:
    fig = keyword_plot(KEYWORDS, "token", "LL", **kwargs)
    assert fig.layout[axis].autorange == expected


@pytest.mark.parametrize(
    "reverse,expected", [(False, "middle right"), (True, "middle left")]
)
def test_keyword_plot_labels_follow_the_stems(reverse: bool, expected: str) -> None:
    fig = keyword_plot(KEYWORDS, "token", "LL", reverse=reverse)
    assert markers(fig).textposition == expected


def test_keyword_plot_ignores_other_columns() -> None:
    # A keyword table carries columns the plot never reads, `keywords()`'s
    # frequency struct among them.
    keywords = KEYWORDS.with_columns(
        pl.struct(f12=pl.col("LL"), f1=pl.col("LL")).alias("freqs")
    )
    fig = keyword_plot(keywords, "token", "LL")
    assert list(markers(fig).text) == ["cat", "dog", "eel"]


def test_keyword_plot_empty_keyword_df() -> None:
    # Left to itself the plot would come out with no stems and a bare axis.
    with pytest.raises(ValueError, match="the keyword_df is empty"):
        keyword_plot(KEYWORDS.clear(), "token", "LL")
    with pytest.raises(ValueError, match="the keyword_df is empty"):
        keyword_plot(KEYWORDS.lazy().filter(pl.col("LL") > 100), "token", "LL")


def test_keyword_plot_missing_column() -> None:
    with pytest.raises(ValueError, match="the keyword_df has no column 'lemma'"):
        keyword_plot(KEYWORDS, "lemma", "LL")
