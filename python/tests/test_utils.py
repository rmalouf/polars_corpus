import polars as pl
import pytest
from polars_corpus.utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_columns,
    collect_like,
)

CORPUS = pl.DataFrame({"token": ["a", "b"], "file_id": ["f1", "f1"]})

METHODS = ("ttest", "pmi", "ll", "chisq")


@pytest.mark.parametrize("frame", [CORPUS, CORPUS.lazy()])
def test_as_corpus_accepts_frames(frame: pl.DataFrame | pl.LazyFrame) -> None:
    assert isinstance(as_corpus(frame), pl.LazyFrame)


@pytest.mark.parametrize("bad", ["corpus.csv", None, [1, 2], CORPUS["token"]])
def test_as_corpus_rejects_non_frames(bad: object) -> None:
    with pytest.raises(ValueError, match="the target corpus must be a polars"):
        as_corpus(bad, "target corpus")


def test_as_corpus_rejects_empty() -> None:
    with pytest.raises(ValueError, match="the corpus is empty"):
        as_corpus(CORPUS.clear())


def test_as_corpus_allows_empty_lazyframe() -> None:
    # Height is unknown without running the query, so an empty LazyFrame passes.
    assert isinstance(as_corpus(CORPUS.clear().lazy()), pl.LazyFrame)


@pytest.mark.parametrize(
    "source,expected",
    [(CORPUS, pl.DataFrame), (CORPUS.lazy(), pl.LazyFrame)],
)
def test_collect_like_follows_the_input(source: object, expected: type) -> None:
    result = collect_like(CORPUS.lazy().select("token"), source)
    assert isinstance(result, expected)


def test_as_corpus_and_collect_like_round_trip() -> None:
    # The pattern in full: take either kind of frame, work lazily, give back
    # what the caller passed.
    for source in (CORPUS, CORPUS.lazy()):
        lazy = as_corpus(source).select("token")
        assert type(collect_like(lazy, source)) is type(source)


@pytest.mark.parametrize("frame", [CORPUS, CORPUS.lazy()])
def test_check_columns_passes(frame: pl.DataFrame | pl.LazyFrame) -> None:
    check_columns(frame, ["token", "file_id"])


def test_check_columns_reports_missing_and_available() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_columns(CORPUS, ["lemma"], "reference corpus")
    assert "the reference corpus has no column 'lemma'" in str(excinfo.value)
    assert "its columns are: token, file_id" in str(excinfo.value)


def test_check_columns_names_the_parameter_to_change() -> None:
    with pytest.raises(ValueError, match="Use file_id_column= to point at"):
        check_columns(CORPUS, ["text_id"], param="file_id_column")


@pytest.mark.parametrize("expr", ["token", pl.col("token")])
def test_as_expr_accepts_names_and_expressions(expr: object) -> None:
    assert as_expr(expr).meta.output_name() == "token"


def test_as_expr_rejects_series_with_its_name() -> None:
    with pytest.raises(ValueError, match="not a Series.* e.g. 'token'"):
        as_expr(CORPUS["token"], hint=" Because reasons.")


@pytest.mark.parametrize("bad", [3, None, ["token"]])
def test_as_expr_rejects_other_types(bad: object) -> None:
    with pytest.raises(ValueError, match="term must be a column name"):
        as_expr(bad, param="term")


@pytest.mark.parametrize("value", ["ll", "LL", " ll "])
def test_check_choice_normalizes(value: str) -> None:
    assert check_choice(value, METHODS) == "ll"


@pytest.mark.parametrize(
    "value,match",
    [
        ("bogus", "Choose one of: ttest, pmi, ll, chisq"),
        ("t-test", "Did you mean 'ttest'"),
        ("chi2", "Did you mean 'chisq'"),
        (None, "got NoneType"),
    ],
)
def test_check_choice_rejects(value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_choice(value, METHODS)


def test_check_choice_names_the_parameter() -> None:
    with pytest.raises(ValueError, match="Unknown syntax 'bogus'"):
        check_choice("bogus", ("simple", "cqp"), param="syntax")
