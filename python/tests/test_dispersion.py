from itertools import combinations
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


def expected(counts: list[float], n_files: int, method: str, size: float = 1) -> float:
    """The measure, computed the long way round from per-file relative frequencies."""
    padded = [c / size for c in counts] + [0.0] * (n_files - len(counts))
    mean = sum(padded) / n_files
    sd = sqrt(sum((x - mean) ** 2 for x in padded) / n_files)
    cv = sd / mean
    # Every pair of files, spelled out, rather than the closed form DA uses.
    pairs = [abs(a - b) for a, b in combinations(padded, 2)]
    return {
        "range": sum(x > 0 for x in padded),
        "range%": 100 * sum(x > 0 for x in padded) / n_files,
        "sd": sd,
        "cv": cv,
        "cv%": cv * 100,
        "d": 1 - cv / sqrt(n_files - 1),
        "da": 1 - (sum(pairs) / len(pairs)) / (2 * mean),
        # Equal-sized files, so each is expected to hold a share 1 / n_files.
        "dp": sum(abs(x / sum(padded) - 1 / n_files) for x in padded) / 2,
    }[method]


@pytest.mark.parametrize(
    "method,column",
    [
        ("range", "range"),
        ("range%", "range%"),
        ("sd", "sd"),
        ("cv", "cv"),
        ("cv%", "cv%"),
        ("d", "D"),
        ("da", "DA"),
        ("dp", "DP"),
    ],
)
def test_dispersion_values(method: str, column: str) -> None:
    result = dispersion(CORPUS, "token", method)

    assert result.columns == ["token", column]
    counts = {"a": [2.0, 1.0, 3.0], "b": [2.0, 1.0], "c": [1.0, 3.0], "d": [2.0]}
    got = dict(zip(result["token"], result[column]))
    assert got == pytest.approx(
        {w: expected(c, 3, method, size=5) for w, c in counts.items()}
    )


@pytest.mark.parametrize(
    "method,column", [("cv", "cv"), ("d", "D"), ("da", "DA"), ("dp", "DP")]
)
def test_dispersion_ranks_by_evenness(method: str, column: str) -> None:
    # dispersion() returns rows unordered, so rank them here. An even spread is
    # the low end for the measures of unevenness and the high end for D and DA.
    result = dispersion(CORPUS, "token", method).sort(
        column, descending=method in ("d", "da")
    )
    # "a" is spread over all three files; "d" occurs in one file only.
    assert result["token"].to_list() == ["a", "b", "c", "d"]


@pytest.mark.parametrize(
    "method,column,counts,expected_value",
    [
        # Perfectly even, then confined to a single file, for each measure.
        ("d", "D", [1, 1, 1], 1.0),
        ("d", "D", [3, 0, 0], 0.0),
        ("da", "DA", [1, 1, 1], 1.0),
        ("da", "DA", [3, 0, 0], 0.0),
        # DP runs the other way, and stops short of 1 by the file's own share.
        ("dp", "DP", [1, 1, 1], 0.0),
        ("dp", "DP", [3, 0, 0], 2 / 3),
    ],
)
def test_dispersion_endpoints(
    counts: list[int], expected_value: float, method: str, column: str
) -> None:
    # Pad every file out to the same size, so a word's dispersion depends only
    # on how it is spread, not on incidental differences in file length.
    size = 3
    corpus = pl.DataFrame(
        {
            "token": ["x"] * sum(counts) + ["pad"] * (size * len(counts) - sum(counts)),
            "file_id": [f"f{i}" for i, n in enumerate(counts) for _ in range(n)]
            + [f"f{i}" for i, n in enumerate(counts) for _ in range(size - n)],
        }
    )
    result = dispersion(corpus, "token", method)
    got = dict(zip(result["token"], result[column]))
    assert got["x"] == pytest.approx(expected_value)


@pytest.mark.parametrize("method,column,even", [("d", "D", 1.0), ("dp", "DP", 0.0)])
def test_dispersion_divides_out_file_length(
    method: str, column: str, even: float
) -> None:
    # Same rate in both files, but f1 is three times the length of f2. Measured
    # on relative frequencies rather than raw counts, this reads as even.
    corpus = pl.DataFrame(
        {
            "token": ["x", "x", "x", "y", "y", "y", "x", "y"],
            "file_id": ["f1"] * 6 + ["f2"] * 2,
        }
    )
    result = dispersion(corpus, "token", method)

    assert result[column].to_list() == pytest.approx([even, even])


@pytest.mark.parametrize("method,column", [("d", "D"), ("da", "DA")])
def test_dispersion_single_file_is_undefined(method: str, column: str) -> None:
    corpus = pl.DataFrame({"token": ["a", "b"], "file_id": ["f1", "f1"]})
    result = dispersion(corpus, "token", method)
    assert all(isnan(value) for value in result[column])


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


@pytest.mark.parametrize(
    "method,column,value",
    [("range", "range", 1), ("range%", "range%", 100.0), ("dp", "DP", 0.0)],
)
def test_dispersion_single_file_defined(method: str, column: str, value: float) -> None:
    # Unlike D and DA, these are still defined for a corpus of one file.
    corpus = pl.DataFrame({"token": ["a", "b"], "file_id": ["f1", "f1"]})
    result = dispersion(corpus, "token", method)
    assert result[column].to_list() == [value, value]


@pytest.mark.parametrize("method", ["range", "sd", "d", "da", "dp"])
def test_dispersion_string_term_matches_expr(method: str) -> None:
    from_str = dispersion(CORPUS, "token", method)
    from_expr = dispersion(CORPUS, pl.col("token"), method)
    assert_frame_equal(from_str, from_expr, check_row_order=False)


@pytest.mark.parametrize(
    "method", ["range", "range%", "sd", "cv", "cv%", "d", "da", "dp"]
)
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


@pytest.mark.parametrize("method", ["D", " CV% ", "Sd", "Da", "DP"])
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
