import warnings

import polars as pl
import pytest

# Plotting is the "examples" extra, not a core dependency.
pytest.importorskip("seaborn")
matplotlib = pytest.importorskip("matplotlib")
# Draw to a buffer: these tests read the axes back rather than showing them.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from polars_corpus import barcode_plot, dispersion_plot, keyword_plot

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


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def y_labels(ax) -> list[str]:
    return [label.get_text() for label in ax.get_yticklabels()]


def test_barcode_plot_rows_follow_targets() -> None:
    # Row order is the caller's, not whichever word the corpus happens to hit first.
    ax = barcode_plot(CORPUS, "token", ["the", "cat"])
    assert y_labels(ax) == ["the", "cat"]


def test_barcode_plot_accepts_a_single_target() -> None:
    ax = barcode_plot(CORPUS, "token", "cat")
    assert y_labels(ax) == ["cat"]


def test_barcode_plot_absent_target_warns_and_keeps_its_row() -> None:
    # An absent word is worth seeing as an empty row: that is the finding.
    with pytest.warns(UserWarning, match="'eel' does not occur"):
        ax = barcode_plot(CORPUS, "token", ["cat", "eel"])
    assert y_labels(ax) == ["cat", "eel"]


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
    ax = barcode_plot(corpus, pl.col("token").str.to_lowercase().alias("norm"), "cat")
    assert y_labels(ax) == ["cat"]
    # Both spellings are drawn, at their positions in the corpus.
    assert [x for c in ax.collections for x, _ in c.get_offsets()] == [0, 1]


def test_dispersion_plot_rows_are_the_files_hit() -> None:
    ax = dispersion_plot(CORPUS, "token", "dog")
    # "dog" is in f1 only, so f2 gets no row.
    assert y_labels(ax) == ["f1"]
    assert y_labels(dispersion_plot(CORPUS, "token", "cat")) == ["f1", "f2"]


@pytest.mark.parametrize(
    "relative,expected", [(True, [0.25, 0.5]), (False, [1.0, 2.0])]
)
def test_dispersion_plot_positions(relative: bool, expected: list[float]) -> None:
    # "cat" is the 1st token of f1 and the 2nd of f2, each four tokens long, so
    # relative positions are counted within the file rather than across the corpus.
    ax = dispersion_plot(CORPUS, "token", "cat", relative=relative)
    got = sorted(
        x for collection in ax.collections for x, _ in collection.get_offsets()
    )
    assert got == pytest.approx(expected)


def test_dispersion_plot_relative_index_is_a_fraction() -> None:
    ax = dispersion_plot(CORPUS, "token", "cat", relative=True)
    xs = [x for collection in ax.collections for x, _ in collection.get_offsets()]
    assert all(0 < x <= 1 for x in xs)
    assert ax.get_xlabel() == "relative index"


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
    ax = dispersion_plot(corpus, term, "cat")
    assert y_labels(ax) == ["f1", "f2"]


def test_dispersion_plot_file_id_column() -> None:
    ax = dispersion_plot(
        CORPUS.rename({"file_id": "text_id"}), "token", "cat", file_id_column="text_id"
    )
    assert y_labels(ax) == ["f1", "f2"]


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
    ax = keyword_plot(KEYWORDS, "token", "LL")
    assert [text.get_text() for text in ax.texts] == ["cat", "dog", "eel"]


def test_keyword_plot_top_k() -> None:
    ax = keyword_plot(KEYWORDS, "token", "LL", top_k=2)
    assert [text.get_text() for text in ax.texts] == ["cat", "dog"]


def test_keyword_plot_computed_term() -> None:
    # A computed term exists only in the select, so the labels have to be read
    # from there rather than from the frame the caller handed over.
    ax = keyword_plot(KEYWORDS, pl.col("token").str.to_uppercase(), "LL")
    assert [text.get_text() for text in ax.texts] == ["CAT", "DOG", "EEL"]


def test_keyword_plot_computed_keyness() -> None:
    ax = keyword_plot(KEYWORDS, "token", (pl.col("LL") * 2).alias("scaled"))
    assert list(ax.containers[0].markerline.get_xdata()) == [18.0, 8.0, 2.0]


def test_keyword_plot_ignores_other_columns() -> None:
    # A keyword table carries columns the plot never reads, `keywords()`'s
    # frequency struct among them.
    keywords = KEYWORDS.with_columns(
        pl.struct(f12=pl.col("LL"), f1=pl.col("LL")).alias("freqs")
    )
    assert [
        text.get_text() for text in keyword_plot(keywords, "token", "LL").texts
    ] == [
        "cat",
        "dog",
        "eel",
    ]


def test_keyword_plot_empty_keyword_df() -> None:
    # Left to itself the stem plot raises out of numpy, reducing over no data.
    with pytest.raises(ValueError, match="the keyword_df is empty"):
        keyword_plot(KEYWORDS.clear(), "token", "LL")
    with pytest.raises(ValueError, match="the keyword_df is empty"):
        keyword_plot(KEYWORDS.lazy().filter(pl.col("LL") > 100), "token", "LL")


def test_keyword_plot_missing_column() -> None:
    with pytest.raises(ValueError, match="the keyword_df has no column 'lemma'"):
        keyword_plot(KEYWORDS, "lemma", "LL")
