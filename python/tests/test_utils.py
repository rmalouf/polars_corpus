import polars as pl
import pytest
from polars_corpus.utils import (
    as_corpus,
    as_expr,
    check_choice,
    check_choices,
    check_columns,
    check_expr,
    collect_like,
    is_letters,
    proportion,
)

CORPUS = pl.DataFrame({"token": ["a", "b"], "file_id": ["f1", "f1"]})

METHODS = ("ttest", "pmi", "ll", "chisq")

# Counts of one word in four files of two genres, one file not counted yet.
COUNTS = pl.DataFrame(
    {
        "file_id": ["f1", "f2", "f3", "f4"],
        "file_type": ["news", "news", "fiction", "fiction"],
        "count": [1, None, 3, 4],
    }
)


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


@pytest.mark.parametrize("frame", [CORPUS, CORPUS.lazy()])
@pytest.mark.parametrize(
    "expr,expected",
    [
        (pl.col("token"), "token"),
        (pl.col("token").str.to_uppercase(), "token"),  # keeps its root name
        (pl.col("token").alias("word"), "word"),
        (pl.col("^tok.*$"), "token"),  # names its column by pattern
        (pl.first(), "token"),  # names it by position
        (pl.col("token") + pl.col("file_id"), "token"),  # more than one root
    ],
)
def test_check_expr_resolves_the_output_column(
    frame: pl.DataFrame | pl.LazyFrame, expr: pl.Expr, expected: str
) -> None:
    assert check_expr(frame, expr) == expected


def test_check_expr_reports_a_missing_column_like_check_columns() -> None:
    with pytest.raises(ValueError) as excinfo:
        check_expr(CORPUS, pl.col("lemma").str.to_uppercase(), "reference corpus")
    assert "the reference corpus has no column 'lemma'" in str(excinfo.value)
    assert "its columns are: token, file_id" in str(excinfo.value)


@pytest.mark.parametrize(
    "expr,match",
    [
        (pl.col("token", "file_id"), "it selects token, file_id"),
        (pl.exclude("token", "file_id"), "it selects none"),
    ],
)
def test_check_expr_rejects_expressions_naming_other_than_one_column(
    expr: pl.Expr, match: str
) -> None:
    with pytest.raises(
        ValueError, match=f"term must identify a single column.*{match}"
    ):
        check_expr(CORPUS, expr, param="term")


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


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ll", ["ll"]),  # one option is a list of one
        (" LL ", ["ll"]),  # normalized as check_choice normalizes it
        (["pmi", "ll"], ["pmi", "ll"]),  # kept in the order asked for
        (("ll", "pmi"), ["ll", "pmi"]),  # a tuple will do
        (["ll", "LL", "pmi"], ["ll", "pmi"]),  # repeats dropped, first wins
    ],
)
def test_check_choices_normalizes(value: object, expected: list[str]) -> None:
    assert check_choices(value, METHODS) == expected


@pytest.mark.parametrize(
    "value,match",
    [
        ([], "method is empty"),
        (["ll", "bogus"], "Unknown method 'bogus'"),
        (3, "or a list of them; got int"),
        ({"ll"}, "or a list of them; got set"),
        ([None], "got NoneType"),
    ],
)
def test_check_choices_rejects(value: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_choices(value, METHODS)


@pytest.mark.parametrize("expr", ["count", pl.col("count")])
def test_proportion_takes_a_name_or_an_expression(expr: object) -> None:
    result = COUNTS.select(proportion(expr))
    # The counts total 8, the null neither counted nor filled in.
    assert result["count"].to_list() == [0.125, None, 0.375, 0.5]


def test_proportion_keeps_the_name_of_the_column_it_reads() -> None:
    assert COUNTS.select(proportion("count")).columns == ["count"]


def test_proportion_shares_sum_to_one() -> None:
    assert COUNTS.select(proportion("count").sum()).item() == pytest.approx(1.0)


def test_proportion_scales_to_a_basis() -> None:
    result = COUNTS.select(proportion("count") * 10_000)
    assert result["count"].to_list() == [1_250.0, None, 3_750.0, 5_000.0]


@pytest.mark.parametrize("group_by", ["file_type", pl.col("file_type")])
def test_proportion_takes_the_total_within_a_group(group_by: object) -> None:
    result = COUNTS.select(proportion("count", group_by))
    # Each count over its own genre's total: 1 of 1, then 3 and 4 of 7.
    assert result["count"].to_list() == [1.0, None, pytest.approx(3 / 7), 4 / 7]


def test_proportion_reads_a_derived_column() -> None:
    result = COUNTS.select(proportion(pl.col("count").fill_null(0)))
    assert result["count"].to_list() == [0.125, 0.0, 0.375, 0.5]


@pytest.mark.parametrize(
    "token,expected",
    [
        ("cat", True),
        ("The", True),  # case is nothing to do with it
        ("naïve", True),  # a precomposed accent is one letter
        ("étude", True),  # and so is a letter with a combining mark
        ("́", False),  # but a mark with no letter to follow is not one
        ("Ελλάδα", True),  # any script, not [A-Za-z]
        ("日本語", True),
        ("don't", True),  # an apostrophe among letters is a letter
        ("don’t", True),  # typographic or not
        ("co-op", True),  # as is a hyphen
        ("'tis", True),  # at either end
        ("pre-", True),
        ("-", False),  # but punctuation alone is punctuation
        ("'", False),
        ("--", False),
        ("3rd", False),  # a digit is not a letter
        ("42", False),
        ("!", False),
        ("two words", False),  # a space is not a letter either
        ("", False),  # one letter at least
        (None, None),
    ],
)
def test_is_letters(token: str | None, expected: bool | None) -> None:
    frame = pl.DataFrame({"token": [token]}, schema={"token": pl.Utf8})
    assert frame.select(is_letters("token")).item() is expected


@pytest.mark.parametrize(
    "token,expected",
    [
        ("two words", True),  # a space is a letter now
        ("New York City", True),
        ("cat", True),  # and a single word still is one
        ("don't stop", True),  # apostrophes and hyphens keep working
        ("well-known fact", True),
        (" cat ", True),  # at either end, as a hyphen would be
        (" ", False),  # but a space alone is still not a letter
        ("two words!", False),
        ("3 words", False),
        ("", False),
        (None, None),
    ],
)
def test_is_letters_allows_spaces(token: str | None, expected: bool | None) -> None:
    frame = pl.DataFrame({"token": [token]}, schema={"token": pl.Utf8})
    assert frame.select(is_letters("token", allow_spaces=True)).item() is expected


def test_is_letters_reads_a_derived_column() -> None:
    frame = pl.DataFrame({"token": ["Cat", "3rd"]})
    result = frame.select(is_letters(pl.col("token").str.to_lowercase()))
    assert result["token"].to_list() == [True, False]
