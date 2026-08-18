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

    assert result.columns == ["token", "freq", column]
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

    assert result.columns == ["norm", "freq", "D"]
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


ALL_METHODS = ["range", "range%", "sd", "cv", "cv%", "d", "da", "dp"]

# Corpus frequencies of CORPUS, summed over the per-file counts above.
FREQS = {"a": 6, "b": 3, "c": 4, "d": 2}


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dispersion_reports_frequency(method: str) -> None:
    # Every method reports the same frequency, whichever route it takes to it.
    result = dispersion(CORPUS, "token", method)
    assert dict(zip(result["token"], result["freq"])) == FREQS


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize(
    "min_freq,expected_words",
    [
        (0, {"a", "b", "c", "d"}),  # the default keeps everything
        (3, {"a", "b", "c"}),  # "d" (freq 2) drops out
        (4, {"a", "c"}),  # inclusive bound: "c" (freq 4) survives
        (7, set()),  # above every word in the corpus
    ],
)
def test_dispersion_min_freq(
    method: str, min_freq: int, expected_words: set[str]
) -> None:
    result = dispersion(CORPUS, "token", method, min_freq=min_freq)
    assert set(result["token"]) == expected_words


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dispersion_min_freq_only_filters(method: str) -> None:
    # Filtering happens after the measure, so the words that survive keep the
    # values they had with the rest of the corpus in view.
    full = dispersion(CORPUS, "token", method).filter(pl.col("freq") >= 3)
    got = dispersion(CORPUS, "token", method, min_freq=3)
    assert_frame_equal(full, got, check_row_order=False)


def test_dispersion_freq_counts_surviving_rows() -> None:
    # Nulls are dropped before anything is counted, so they are not in `freq`.
    corpus = pl.concat(
        [CORPUS, pl.DataFrame({"token": ["a", "a"], "file_id": [None, None]})]
    )
    result = dispersion(corpus, "token", "d")
    assert dict(zip(result["token"], result["freq"])) == FREQS


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dispersion_drops_null_file_ids(method: str) -> None:
    # A token with no file id belongs to no part, so it cannot be spread across
    # them. Every method has to agree about that -- 'range' used to count null
    # as a file of its own while the rest silently dropped the word.
    corpus = CORPUS.with_columns(
        pl.when(pl.col("file_id") == "f3")
        .then(None)
        .otherwise(pl.col("file_id"))
        .alias("file_id")
    )
    got = dispersion(corpus, "token", method)
    expected_result = dispersion(
        CORPUS.filter(pl.col("file_id") != "f3"), "token", method
    )
    assert_frame_equal(got, expected_result, check_row_order=False)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_dispersion_drops_null_terms(method: str) -> None:
    # A null term is not an occurrence of anything, so it gets no row of its own
    # -- and the file sizes the measures divide by count only what survives.
    corpus = pl.concat(
        [CORPUS, pl.DataFrame({"token": [None] * 3, "file_id": ["f1", "f2", "f3"]})]
    )
    got = dispersion(corpus, "token", method)
    assert None not in got["token"].to_list()
    assert_frame_equal(got, dispersion(CORPUS, "token", method), check_row_order=False)


def test_dispersion_lazy_drops_nulls() -> None:
    # The lazy path drops on the same terms as the eager one.
    corpus = pl.concat(
        [CORPUS, pl.DataFrame({"token": [None], "file_id": ["f1"]})]
    ).lazy()
    got = dispersion(corpus, "token", "d").collect()
    assert None not in got["token"].to_list()


def test_dispersion_all_rows_null() -> None:
    corpus = pl.DataFrame({"token": [None, None], "file_id": ["f1", "f2"]})
    assert dispersion(corpus, "token", "d").height == 0


COLUMNS = {
    "range": "range",
    "range%": "range%",
    "sd": "sd",
    "cv": "cv",
    "cv%": "cv%",
    "d": "D",
    "da": "DA",
    "dp": "DP",
}


@pytest.mark.parametrize(
    "methods",
    [
        ALL_METHODS,  # every group at once
        ["range", "d"],  # two groups
        ["sd", "cv", "cv%", "d"],  # one group, all of it
        ["da", "dp"],  # the two that stand alone
        ["dp"],  # a list of one is still a list
    ],
)
def test_dispersion_several_methods(methods: list[str]) -> None:
    got = dispersion(CORPUS, "token", methods)

    # Reported in the order asked for, whatever order they are computed in.
    assert got.columns == ["token", "freq"] + [COLUMNS[m] for m in methods]
    # And each column holds what asking for that measure alone would have given.
    for method in methods:
        alone = dispersion(CORPUS, "token", method)
        assert_frame_equal(
            got.select("token", "freq", COLUMNS[method]),
            alone,
            check_row_order=False,
        )


def test_dispersion_several_methods_keeps_the_order_asked_for() -> None:
    # Not the order of METHODS, and not the order the passes run in.
    got = dispersion(CORPUS, "token", ["dp", "range", "d"])
    assert got.columns == ["token", "freq", "DP", "range", "D"]


def test_dispersion_repeated_method() -> None:
    # Two columns of the same name would not be a frame; drop the repeat.
    got = dispersion(CORPUS, "token", ["d", "range", "d"])
    assert got.columns == ["token", "freq", "D", "range"]


@pytest.mark.parametrize("methods", [["range", "d"], ALL_METHODS])
def test_dispersion_several_methods_min_freq(methods: list[str]) -> None:
    got = dispersion(CORPUS, "token", methods, min_freq=3)
    assert set(got["token"]) == {"a", "b", "c"}


@pytest.mark.parametrize("methods", [["range", "d"], ["da", "dp"]])
def test_dispersion_several_methods_lazy_matches_eager(methods: list[str]) -> None:
    eager = dispersion(CORPUS, "token", methods)
    lazy = dispersion(CORPUS.lazy(), "token", methods).collect()
    assert_frame_equal(eager, lazy, check_row_order=False)


def test_dispersion_several_methods_case_insensitive() -> None:
    got = dispersion(CORPUS, "token", [" D ", "Range%"])
    assert got.columns == ["token", "freq", "D", "range%"]


def test_dispersion_invalid_method() -> None:
    with pytest.raises(ValueError, match="sd, cv, cv%, d"):
        dispersion(CORPUS, "token", "bogus")


def test_dispersion_invalid_method_in_a_list() -> None:
    with pytest.raises(ValueError, match="Unknown method 'bogus'"):
        dispersion(CORPUS, "token", ["d", "bogus"])


def test_dispersion_empty_method_list() -> None:
    with pytest.raises(ValueError, match="method is empty"):
        dispersion(CORPUS, "token", [])


@pytest.mark.parametrize("bad", [3, None, {"d"}])
def test_dispersion_method_wrong_type(bad: object) -> None:
    with pytest.raises(ValueError, match="or a list of them"):
        dispersion(CORPUS, "token", bad)


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
