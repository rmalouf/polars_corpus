import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_corpus import frequency_list, is_letters

from .helpers import corpus

# Three files, eleven tokens. Counts, as raw forms:
#   cat : 3 (all three files)     "." : 2 (f1, f3)
#   The : 1   the : 1             sat/ran/dogs/"!" : 1 each
# Folding case merges The with the; keeping only letters drops the two "."
# and the one "!", leaving eight tokens.
CORPUS = pl.DataFrame(
    {
        "token": [
            "The",
            "cat",
            "sat",
            ".",
            "the",
            "cat",
            "ran",
            "!",
            "dogs",
            "cat",
            ".",
        ],
        "file_id": ["f1"] * 4 + ["f2"] * 4 + ["f3"] * 3,
    }
)

RAW = {"cat": 3, ".": 2, "!": 1, "The": 1, "dogs": 1, "ran": 1, "sat": 1, "the": 1}
FOLDED = {"cat": 3, ".": 2, "the": 2, "!": 1, "dogs": 1, "ran": 1, "sat": 1}
LETTERS = {"cat": 3, "The": 1, "dogs": 1, "ran": 1, "sat": 1, "the": 1}
BOTH = {"cat": 3, "the": 2, "dogs": 1, "ran": 1, "sat": 1}

# Files each raw form occurs in.
RANGE = {"cat": 3, ".": 2, "!": 1, "The": 1, "dogs": 1, "ran": 1, "sat": 1, "the": 1}


# Normalizing is expr's job, so each of these is the call a caller writes.
NORMALIZED = {
    "raw": (CORPUS, "token", RAW),
    "lowercase": (CORPUS, pl.col("token").str.to_lowercase(), FOLDED),
    "letters_only": (CORPUS.filter(is_letters("token")), "token", LETTERS),
    "both": (
        CORPUS.filter(is_letters("token")),
        pl.col("token").str.to_lowercase(),
        BOTH,
    ),
    # A null from expr is dropped, so when/then restricts without a filter.
    "when_then": (
        CORPUS,
        pl.when(is_letters("token"))
        .then(pl.col("token").str.to_lowercase())
        .alias("token"),
        BOTH,
    ),
}


@pytest.mark.parametrize("case", list(NORMALIZED), ids=list(NORMALIZED))
def test_counts(case: str) -> None:
    frame, expr, expected = NORMALIZED[case]
    result = frequency_list(frame, expr)

    assert result.columns == ["token", "freq", "rate", "range"]
    assert dict(zip(result["token"], result["freq"])) == expected


@pytest.mark.parametrize("case", ["raw", "both", "when_then"])
def test_rate_is_the_share_of_the_tokens_counted(case: str) -> None:
    frame, expr, expected = NORMALIZED[case]
    total = sum(expected.values())
    result = frequency_list(frame, expr, basis=10_000)

    rates = dict(zip(result["token"], result["rate"]))
    assert rates == pytest.approx({w: 10_000 * c / total for w, c in expected.items()})
    assert result["rate"].sum() == pytest.approx(10_000)


@pytest.mark.parametrize("basis", [1, 100, 10_000, 1_000_000])
def test_rate_scales_with_basis(basis: float) -> None:
    result = frequency_list(CORPUS, "token", basis=basis)

    assert result["rate"].sum() == pytest.approx(basis)
    assert result["rate"][0] == pytest.approx(basis * 3 / 11)


def test_range_counts_distinct_files() -> None:
    result = frequency_list(CORPUS, "token")

    assert dict(zip(result["token"], result["range"])) == RANGE


def test_sorted_by_frequency_then_by_word() -> None:
    result = frequency_list(CORPUS, "token")

    # Descending count; within a count, the word ascending.
    assert result["token"].to_list() == sorted(RAW, key=lambda w: (-RAW[w], w))


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (pl.col("freq") >= 2, ["cat", "."]),
        (pl.col("freq") >= 3, ["cat"]),
        (pl.col("range") >= 2, ["cat", "."]),
        (pl.col("range") >= 3, ["cat"]),
        ((pl.col("freq") >= 2) & (pl.col("range") >= 3), ["cat"]),
    ],
)
def test_thresholding_is_a_filter_on_the_result(
    predicate: pl.Expr, expected: list[str]
) -> None:
    result = frequency_list(CORPUS, "token").filter(predicate)

    assert result["token"].to_list() == sorted(expected, key=lambda w: (-RAW[w], w))


@pytest.mark.parametrize("predicate", [pl.col("freq") >= 3, pl.col("range") >= 3])
def test_filtering_the_result_leaves_the_rate_alone(predicate: pl.Expr) -> None:
    # The rate divides by the tokens counted, not by the rows a later filter
    # keeps, so "cat" keeps the rate it had in the whole corpus.
    everything = frequency_list(CORPUS, "token", basis=10_000)
    filtered = everything.filter(predicate)

    assert filtered["rate"].to_list() == pytest.approx([everything["rate"][0]])
    assert filtered["rate"][0] == pytest.approx(10_000 * 3 / 11)


def test_expr_names_the_output_column() -> None:
    result = frequency_list(CORPUS, pl.col("token").str.to_lowercase())

    assert result.columns[0] == "token"
    assert dict(zip(result["token"], result["freq"])) == FOLDED


def test_expr_defaults_to_the_token_column() -> None:
    assert_frame_equal(frequency_list(CORPUS), frequency_list(CORPUS, "token"))


def test_struct_expr_counts_the_pair() -> None:
    df = corpus(token="a b a", pos="N V V", file_id="f f g")
    result = frequency_list(df, pl.struct("token", "pos"))

    assert result["freq"].to_list() == [1, 1, 1]
    assert result.columns == ["token", "freq", "rate", "range"]


@pytest.mark.parametrize("lazy", [False, True], ids=["eager", "lazy"])
def test_returns_the_frame_type_it_was_given(lazy: bool) -> None:
    given = CORPUS.lazy() if lazy else CORPUS
    result = frequency_list(given, "token")

    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    frame = result.collect() if lazy else result
    assert_frame_equal(frame, frequency_list(CORPUS, "token"))


def test_namespace_matches_the_function() -> None:
    assert_frame_equal(CORPUS.corpus.frequency_list("token"), frequency_list(CORPUS))
    assert_frame_equal(
        CORPUS.lazy().corpus.frequency_list("token").collect(),
        frequency_list(CORPUS),
    )


@pytest.mark.parametrize(
    "kwargs", [{}, {"file_id_column": None}], ids=["absent", "declined"]
)
def test_no_range_column_without_file_ids(kwargs: dict) -> None:
    df = CORPUS.drop("file_id") if not kwargs else CORPUS
    result = frequency_list(df, "token", **kwargs)

    assert result.columns == ["token", "freq", "rate"]
    assert dict(zip(result["token"], result["freq"])) == RAW


def test_file_id_column_can_be_named() -> None:
    df = CORPUS.rename({"file_id": "text_id"})
    result = frequency_list(df, "token", file_id_column="text_id")

    assert dict(zip(result["token"], result["range"])) == RANGE


def test_nulls_are_dropped() -> None:
    df = pl.DataFrame(
        {"token": ["a", None, "a", "b"], "file_id": ["f", "f", "g", None]}
    )
    result = frequency_list(df, "token")

    assert dict(zip(result["token"], result["freq"])) == {"a": 2}


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"basis": 0}, "basis must be a positive number"),
        ({"basis": -10}, "basis must be a positive number"),
        ({"basis": "lots"}, "basis must be a positive number"),
        ({"file_id_column": "nope"}, "has no column 'nope'"),
    ],
)
def test_bad_arguments_raise(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        frequency_list(CORPUS, "token", **kwargs)


def test_bad_expr_raises() -> None:
    with pytest.raises(ValueError, match="has no column 'lemma'"):
        frequency_list(CORPUS, "lemma")
