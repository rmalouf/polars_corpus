from math import isnan, sqrt

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_corpus import dispersion

# Three files of five tokens each, so raw counts and relative frequencies rank
# the same way. Per-file counts:
#   a : 2, 1, 3  (spread over all three)
#   b : 2, 1, 0
#   c : 1, 3, 0
#   d : 0, 0, 2  (confined to one file)
CORPUS = pl.DataFrame(
    {
        "token": list("aabbc") + list("abccc") + list("aaadd"),
        "file_id": ["f1"] * 5 + ["f2"] * 5 + ["f3"] * 5,
    }
)


def expected(counts: list[float], n_files: int, method: str) -> float:
    """The measure, computed the long way round from per-file frequencies."""
    padded = counts + [0.0] * (n_files - len(counts))
    mean = sum(padded) / n_files
    sd = sqrt(sum((x - mean) ** 2 for x in padded) / n_files)
    cv = sd / mean
    return {
        "sd": sd,
        "cv": cv,
        "cv%": cv * 100,
        "d": 1 - cv / sqrt(n_files - 1),
    }[method]


@pytest.mark.parametrize(
    "method,column",
    [("sd", "sd"), ("cv", "cv"), ("cv%", "cv%"), ("d", "D")],
)
def test_dispersion_values(method: str, column: str) -> None:
    result = dispersion(CORPUS, "token", method, normalize=False)

    assert result.columns == ["token", column]
    counts = {"a": [2.0, 1.0, 3.0], "b": [2.0, 1.0], "c": [1.0, 3.0], "d": [2.0]}
    got = dict(zip(result["token"], result[column]))
    assert got == pytest.approx({w: expected(c, 3, method) for w, c in counts.items()})


@pytest.mark.parametrize("method,column", [("cv", "cv"), ("d", "D")])
def test_dispersion_ranks_by_evenness(method: str, column: str) -> None:
    # dispersion() returns rows unordered, so rank them here. An even spread is
    # the low end for the measures of variation and the high end for D.
    result = dispersion(CORPUS, "token", method, normalize=False).sort(
        column, descending=method == "d"
    )
    # "a" is spread over all three files; "d" occurs in one file only.
    assert result["token"].to_list() == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "counts,expected_d",
    [
        ([1, 1, 1], 1.0),  # perfectly even
        ([3, 0, 0], 0.0),  # confined to a single file
    ],
)
def test_dispersion_d_endpoints(counts: list[int], expected_d: float) -> None:
    corpus = pl.DataFrame(
        {
            "token": ["x"] * sum(counts) + ["pad"] * 3,
            "file_id": [f"f{i}" for i, n in enumerate(counts) for _ in range(n)]
            + ["f0", "f1", "f2"],
        }
    )
    result = dispersion(corpus, "token", "d", normalize=False)
    assert dict(zip(result["token"], result["D"]))["x"] == pytest.approx(expected_d)


def test_dispersion_normalize_divides_out_file_length() -> None:
    # Same rate in both files, but f1 is three times the length of f2. On raw
    # counts that reads as uneven; on relative frequencies it is perfectly even.
    corpus = pl.DataFrame(
        {
            "token": ["x", "x", "x", "y", "y", "y", "x", "y"],
            "file_id": ["f1"] * 6 + ["f2"] * 2,
        }
    )
    normalized = dispersion(corpus, "token", "d")
    raw = dispersion(corpus, "token", "d", normalize=False)

    assert normalized["D"].to_list() == pytest.approx([1.0, 1.0])
    assert raw["D"].to_list() == pytest.approx([0.5, 0.5])


def test_dispersion_single_file_is_undefined() -> None:
    corpus = pl.DataFrame({"token": ["a", "b"], "file_id": ["f1", "f1"]})
    result = dispersion(corpus, "token", "d")
    assert all(isnan(value) for value in result["D"])


def test_dispersion_computed_term() -> None:
    corpus = pl.DataFrame(
        {"token": ["Cat", "CAT", "cat", "dog"], "file_id": ["f1", "f2", "f3", "f1"]}
    )
    result = dispersion(
        corpus, pl.col("token").str.to_lowercase().alias("norm"), "d"
    ).sort("D", descending=True)

    assert result.columns == ["norm", "D"]
    # Case-folded, "cat" is in every file and so is more evenly spread than "dog".
    assert result["norm"].to_list() == ["cat", "dog"]


@pytest.mark.parametrize("term", [pl.col("^tok.*$"), pl.first()])
def test_dispersion_indirect_term(term: pl.Expr) -> None:
    # An expression need not name its column outright: the schema settles it.
    expected_result = dispersion(CORPUS, "token", "d")
    got = dispersion(CORPUS, term, "d")
    assert_frame_equal(expected_result, got, check_row_order=False)


def test_dispersion_multi_column_term() -> None:
    with pytest.raises(ValueError, match="expr must identify a single column"):
        dispersion(CORPUS.with_columns(pos=pl.lit("N")), pl.col("token", "pos"), "d")


@pytest.mark.parametrize("method", ["sd", "d"])
def test_dispersion_string_term_matches_expr(method: str) -> None:
    from_str = dispersion(CORPUS, "token", method)
    from_expr = dispersion(CORPUS, pl.col("token"), method)
    assert_frame_equal(from_str, from_expr, check_row_order=False)


@pytest.mark.parametrize("method", ["sd", "cv", "cv%", "d"])
def test_dispersion_lazy_matches_eager(method: str) -> None:
    eager = dispersion(CORPUS, "token", method)
    lazy = dispersion(CORPUS.lazy(), "token", method).collect()
    assert_frame_equal(eager, lazy, check_row_order=False)


def test_dispersion_file_id_column() -> None:
    expected_result = dispersion(CORPUS, "token", "d")
    got = dispersion(
        CORPUS.rename({"file_id": "text_id"}), "token", "d", file_id_column="text_id"
    )
    assert_frame_equal(expected_result, got, check_row_order=False)


def test_dispersion_ignores_other_columns() -> None:
    # Only the columns `expr` and `file_id_column` name are read.
    expected_result = dispersion(CORPUS, "token", "d")
    got = dispersion(
        CORPUS.with_columns(pos=pl.lit("N"), genre=pl.lit("news")), "token", "d"
    )
    assert_frame_equal(expected_result, got, check_row_order=False)


@pytest.mark.parametrize("method", ["D", " CV% ", "Sd"])
def test_dispersion_method_case_insensitive(method: str) -> None:
    got = dispersion(CORPUS, "token", method)
    expected_result = dispersion(CORPUS, "token", method.strip().lower())
    assert_frame_equal(got, expected_result, check_row_order=False)


def test_dispersion_invalid_method() -> None:
    with pytest.raises(ValueError, match="sd, cv, cv%, d"):
        dispersion(CORPUS, "token", "bogus")


@pytest.mark.parametrize("bad", [[1, 2, 3], "corpus", None, CORPUS["token"]])
def test_dispersion_invalid_corpus(bad: object) -> None:
    with pytest.raises(ValueError, match="the corpus must be a polars"):
        dispersion(bad, "token", "d")


def test_dispersion_empty_corpus() -> None:
    with pytest.raises(ValueError, match="the corpus is empty"):
        dispersion(CORPUS.clear(), "token", "d")


def test_dispersion_missing_term_column() -> None:
    with pytest.raises(ValueError, match="the corpus has no column 'lemma'"):
        dispersion(CORPUS, "lemma", "d")


def test_dispersion_missing_file_id_column() -> None:
    with pytest.raises(ValueError, match="Use file_id_column= to point at"):
        dispersion(CORPUS.drop("file_id"), "token", "d")


def test_dispersion_invalid_term() -> None:
    with pytest.raises(ValueError, match="not a Series"):
        dispersion(CORPUS, CORPUS["token"], "d")
    with pytest.raises(ValueError, match="got int"):
        dispersion(CORPUS, 3, "d")
