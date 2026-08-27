import math

import polars as pl
import pytest
from polars_corpus import collocations, search
from polars_corpus.assoc import (
    chisq,
    logdice,
    loglik,
    mi3,
    minsens,
    pmi,
    tscore,
    zscore,
)
from polars_corpus.collocations import MEASURES as BUILTINS

from .helpers import log_ratio, named_by_alias

# Two files, with "fox" once in each. Positions:
#
#    0    1     2     3   4     5    6   7    8   9  | 10  11    12  13  14
#   the quick brown fox jumps over the lazy dog  .   | the quick red fox runs
#
# The match in f2 sits one token from the end, so a window of 2 reaches only
# one token to its right: any span that runs off a file is cut short there.
CORPUS = pl.DataFrame(
    {
        "token": "the quick brown fox jumps over the lazy dog . "
        "the quick red fox runs".split(),
        "lemma": "the quick brown fox jump over the lazy dog . "
        "the quick red fox run".split(),
        "pos": "DT JJ JJ NN VBZ IN DT JJ NN . DT JJ JJ NN VBZ".split(),
        "sentence": ["B"] + ["I"] * 9 + ["B"] + ["I"] * 4,
        "file_id": ["f1"] * 10 + ["f2"] * 5,
    }
)

FIELDS = [pl.col("freqs").struct.field(name) for name in ("f12", "f1", "f2", "n")]

# Each method with the measure it is meant to report, read back off the freqs
# struct the same result carries. What the formulas themselves compute is
# test_assoc.py's business; this checks the wiring.
MEASURES = {
    "freq": FIELDS[0],
    "pmi": pmi(*FIELDS),
    "mi3": mi3(*FIELDS),
    "logdice": logdice(*FIELDS),
    "ll": loglik(*FIELDS),
    "chisq": chisq(*FIELDS),
    "tscore": tscore(*FIELDS),
    "zscore": zscore(*FIELDS),
    "minsens": minsens(*FIELDS),
}


def fox(corpus: pl.DataFrame | pl.LazyFrame = CORPUS, **kwargs):
    return search(corpus, "fox", **kwargs)


def collocate_freqs(result: pl.DataFrame) -> dict[str, tuple[int, ...]]:
    """The result keyed by collocate, each holding (f12, f1, f2, n)."""
    return {
        row["collocate"]: tuple(row["freqs"].values())
        for row in result.iter_rows(named=True)
    }


def test_counts() -> None:
    result = collocations(fox(), "token", "freq", window=2, min_freq=1)
    freqs = collocate_freqs(result)

    # quick precedes both matches; the rest fall around one or the other.
    assert freqs["quick"] == (2, 7, 2, 15)
    assert freqs["brown"] == (1, 7, 1, 15)
    assert set(freqs) == {"quick", "brown", "jumps", "over", "red", "runs"}
    # f1 counts the context tokens there actually were: four around the first
    # match and three around the second, not 2 x 2 x 2.
    assert sum(f12 for f12, *_ in freqs.values()) == 7


@pytest.mark.parametrize(
    "method,column",
    [(m, expr.meta.output_name()) for m, expr in BUILTINS.items()],
)
def test_method_columns(method: str, column: str) -> None:
    result = collocations(fox(), "token", method, window=2, min_freq=1)

    assert result.columns == ["collocate", "freqs", "range", column]
    expected = result.select(MEASURES[method]).to_series()
    assert result[column].to_list() == pytest.approx(expected.to_list())


def test_methods_together() -> None:
    methods = ["logdice", "freq", "pmi"]
    result = collocations(fox(), "token", methods, window=2, min_freq=1)

    # Columns in the order asked for, ranked by the first of them.
    assert result.columns == ["collocate", "freqs", "range", "LogDice", "freq", "PMI"]
    assert result["LogDice"].to_list() == sorted(result["LogDice"], reverse=True)


@pytest.mark.parametrize(
    "window,expected,f1",
    [
        (2, {"quick", "brown", "jumps", "over", "red", "runs"}, 7),
        ((2, 0), {"quick", "brown", "red"}, 4),
        ((0, 2), {"jumps", "over", "runs"}, 3),
        (1, {"brown", "jumps", "red", "runs"}, 4),
    ],
)
def test_window_shapes(window, expected: set[str], f1: int) -> None:
    result = collocations(fox(), "token", "freq", window=window, min_freq=1)
    freqs = collocate_freqs(result)

    assert set(freqs) == expected
    assert {f for _, f, _, _ in freqs.values()} == {f1}


def test_window_stops_at_file_boundary() -> None:
    # Five to the right of the second match would run into nothing, and five
    # to the left of the first would run off the front of the corpus.
    result = collocations(fox(), "token", "freq", window=5, min_freq=1)
    freqs = collocate_freqs(result)

    assert "." not in freqs  # in f1, but past the end of f2's window
    # "the" at 0 and 6 around the first match, at 10 around the second.
    assert freqs["the"] == (3, 12, 3, 15)


def test_chunk_column_spans_the_sentence() -> None:
    result = collocations(fox(), "token", "freq", chunk_column="sentence", min_freq=1)
    freqs = collocate_freqs(result)

    # Everything in each file but the matched token itself: 9 + 4 tokens.
    assert {f for _, f, _, _ in freqs.values()} == {13}
    assert freqs["."] == (1, 13, 1, 15)


def test_expr_reads_the_column_it_names() -> None:
    result = collocations(fox(), "pos", "freq", window=2, min_freq=1)
    freqs = collocate_freqs(result)

    # Colligation: four adjectives in the windows, out of five in the corpus.
    assert freqs["JJ"] == (4, 7, 5, 15)


def test_expr_may_be_a_struct() -> None:
    result = collocations(fox(), pl.struct("lemma", "pos"), "ll", window=2, min_freq=1)
    collocates = {tuple(row.values()) for row in result["collocate"]}

    assert ("quick", "JJ") in collocates
    assert ("jump", "VBZ") in collocates  # the lemma, not the token "jumps"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"min_freq": 2}, {"quick"}),
        ({"min_freq": 1, "min_range": 2}, {"quick"}),
        (
            {"min_freq": 1, "min_range": 1},
            {"quick", "brown", "jumps", "over", "red", "runs"},
        ),
    ],
)
def test_thresholds(kwargs: dict, expected: set[str]) -> None:
    result = collocations(fox(), "token", "freq", window=2, **kwargs)
    assert set(result["collocate"]) == expected


def test_range_counts_files() -> None:
    result = collocations(fox(), "token", "freq", window=2, min_freq=1)
    ranges = dict(zip(result["collocate"], result["range"]))

    assert ranges["quick"] == 2  # once in each file
    assert ranges["brown"] == 1


def test_lazy_matches_eager() -> None:
    eager = collocations(fox(), "token", "ll", window=2, min_freq=1)
    lazy = collocations(fox(CORPUS.lazy()), "token", "ll", window=2, min_freq=1)

    assert eager.sort("collocate").equals(lazy.sort("collocate"))


def test_no_file_ids_omits_range() -> None:
    result = collocations(
        fox(file_id_column=None), "token", "freq", window=2, min_freq=1
    )
    assert result.columns == ["collocate", "freqs", "freq"]


def test_measures_rank_differently() -> None:
    # The point of offering several: PMI likes the rare word, t-score the
    # frequent one. "sly" occurs once, next to a fox; "the" sits next to a fox
    # six times but is just as common away from one.
    corpus = pl.DataFrame(
        {"token": ("a sly fox . " + "the fox . " * 6 + "the dog . " * 6).split()}
    )
    result = collocations(
        search(corpus, "fox"), "token", ["pmi", "tscore"], window=1, min_freq=1
    )
    by_pmi = result.sort("PMI", descending=True)["collocate"].to_list()
    by_t = result.sort("TScore", descending=True)["collocate"].to_list()

    assert by_pmi.index("sly") < by_pmi.index("the")
    assert by_t.index("the") < by_t.index("sly")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"method": "logdicey"}, "Did you mean 'logdice'"),
        ({"method": []}, "name at least one"),
        ({"window": 0}, "at least one token"),
        ({"window": (0, 0)}, "at least one token"),
        ({"window": (1, 2, 3)}, "got 3 numbers"),
        ({"window": -1}, "non-negative integer"),
        ({"window": 1.5}, "non-negative integer"),
        ({"expr": "nope"}, "no column 'nope'"),
        ({"min_freq": -1}, "min_freq must be"),
        ({"min_range": -1}, "min_range must be"),
    ],
)
def test_bad_arguments(kwargs: dict, message: str) -> None:
    args = {"expr": "token", "method": "ll"} | kwargs
    with pytest.raises(ValueError, match=message):
        collocations(fox(), **args)


# --- Measures of the caller's own ---------------------------------------------

# The eight built-ins that are plain functions of the four counts. 'freq' is a
# bare column reference and has no function form, so it is left out.
BUILTIN_FUNCTIONS = [
    ("pmi", pmi),
    ("mi3", mi3),
    ("logdice", logdice),
    ("ll", loglik),
    ("chisq", chisq),
    ("tscore", tscore),
    ("zscore", zscore),
    ("minsens", minsens),
]


@pytest.mark.parametrize(
    "method,function", BUILTIN_FUNCTIONS, ids=[m for m, _ in BUILTIN_FUNCTIONS]
)
def test_builtin_function_matches_its_name(method: str, function) -> None:
    """Passing a measure's function ranks exactly as naming it does."""
    by_name = collocations(fox(), "token", method, window=2, min_freq=1)
    by_function = collocations(fox(), "token", function, window=2, min_freq=1)

    assert by_function.columns[-1] == function.__name__
    # Keyed by collocate: equal scores tie, and ties do not sort predictably.
    scored = dict(zip(by_function["collocate"], by_function[function.__name__]))
    assert scored == pytest.approx(
        dict(zip(by_name["collocate"], by_name[BUILTINS[method].meta.output_name()]))
    )


@pytest.mark.parametrize(
    "measure,column",
    [(log_ratio, "log_ratio"), (named_by_alias, "LogRatio")],
)
def test_own_measure_names_its_column(measure, column: str) -> None:
    """The function's name, or the alias it puts on what it returns."""
    result = collocations(fox(), "token", measure, window=2, min_freq=1)

    assert result.columns == ["collocate", "freqs", "range", column]
    expected = result.select(measure(*FIELDS)).to_series()
    assert result[column].to_list() == pytest.approx(expected.to_list())


def test_own_measure_beside_builtins() -> None:
    result = collocations(
        fox(), "token", ["ll", log_ratio, "pmi"], window=2, min_freq=1
    )

    # Columns in the order asked for, ranked by the first of them.
    assert result.columns == [
        "collocate",
        "freqs",
        "range",
        "LogLik",
        "log_ratio",
        "PMI",
    ]
    assert result["LogLik"].to_list() == sorted(result["LogLik"], reverse=True)


def test_own_measure_ranks_when_it_comes_first() -> None:
    result = collocations(fox(), "token", [log_ratio, "ll"], window=2, min_freq=1)

    assert result["log_ratio"].to_list() == sorted(result["log_ratio"], reverse=True)


@pytest.mark.parametrize(
    "measure,message",
    [
        (lambda f12, f1, f2, n: f12 / f1, "needs a name for the column"),
        (lambda f12, f1, f2, n: 3, "must return a polars expression"),
        (
            lambda f12, f1, f2, n: (f12 / f1).alias("range"),
            "would be called 'range'",
        ),
        (
            lambda f12, f1, f2, n: (f12 / f1).alias("LogLik"),
            "would be called 'LogLik'",
        ),
    ],
    ids=["unnamed", "not an expression", "reserved column", "clashes with builtin"],
)
def test_bad_own_measure(measure, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        collocations(fox(), "token", ["ll", measure], window=2, min_freq=1)


def test_min_range_needs_file_ids() -> None:
    with pytest.raises(ValueError, match="no file ids"):
        collocations(fox(file_id_column=None), "token", "ll", min_range=2)


@pytest.mark.parametrize(
    "results,message",
    [
        (None, "got None"),
        (CORPUS, "got DataFrame"),
        ("fox", "got str"),
    ],
)
def test_needs_search_results(results, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        collocations(results, "token", "ll")


def test_matches_the_measure_computed_by_hand() -> None:
    # One value end to end, so the plumbing is pinned to a number and not
    # only to the measure it calls: "quick", f12=2, f1=7, f2=2, n=15.
    result = collocations(fox(), "token", "logdice", window=2, min_freq=2)

    assert result["LogDice"].item() == pytest.approx(14 + math.log2(2 * 2 / (7 + 2)))
