import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_corpus import keywords
from polars_corpus.assoc import (
    bic,
    chisq,
    loglik,
    logratio,
    mi3,
    minsens,
    oddsratio,
    pctdiff,
    pmi,
    tscore,
    zscore,
)

from .helpers import jaccard, named_by_alias

# Target corpus. Per-word frequency and range (files it occurs in):
#   cat : freq=3, range=2 (t1, t2)
#   dog : freq=2, range=1 (t1)
#   fish: freq=1, range=1 (t3)
#   the : freq=3, range=3 (t1, t2, t3)
TARGET = pl.DataFrame(
    {
        "norm": ["cat", "cat", "cat", "dog", "dog", "fish", "the", "the", "the"],
        "file_id": ["t1", "t2", "t2", "t1", "t1", "t3", "t1", "t2", "t3"],
    }
)

# Reference corpus: "the" is common here, "dog" appears once, "cat"/"fish" absent.
REFERENCE = pl.DataFrame(
    {
        "norm": ["the", "the", "the", "the", "dog", "water", "water"],
        "file_id": ["r1", "r2", "r3", "r4", "r1", "r2", "r3"],
    }
)

# Expected target ranges (files each word occurs in), keyed by word.
TARGET_RANGE = {"cat": 2, "dog": 1, "fish": 1, "the": 3}


@pytest.mark.parametrize(
    "method,col,kwargs",
    [
        ("ll", "LogLik", {}),
        ("pmi", "PMI", {}),
        ("mi3", "MI3", {}),
        ("chisq", "ChiSq", {}),
        ("tscore", "TScore", {}),
        ("zscore", "ZScore", {}),
        ("smp", "SMP", {"k": 1}),
        ("bic", "BIC", {}),
        ("logratio", "LogRatio", {}),
        ("pctdiff", "%DIFF", {}),
        ("oddsratio", "OddsRatio", {}),
    ],
)
def test_keywords_assoc_structure(method: str, col: str, kwargs: dict) -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), method, **kwargs)

    assert result.columns == ["norm", "freqs", "target_range", col]
    # Only target-corpus words appear, and the part marker is dropped.
    assert set(result["norm"]) == {"cat", "dog", "fish", "the"}
    # Ranked by association strength, descending.
    vals = result[col].to_list()
    assert vals == sorted(vals, reverse=True)


def test_keywords_target_range_values() -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ll")
    got = dict(zip(result["norm"], result["target_range"]))
    assert got == TARGET_RANGE


@pytest.mark.parametrize(
    "term",
    [
        pl.col("token").str.to_lowercase().alias("norm"),  # aliased
        pl.col("token").str.to_lowercase(),  # keeps root name "token"
    ],
)
def test_keywords_computed_term(term: pl.Expr) -> None:
    # A computed term is never materialized on `combined`; group_by evaluates it,
    # and its output name must line up with the crosstab column for target_range.
    target = pl.DataFrame(
        {"token": ["Cat", "CAT", "dog"], "file_id": ["t1", "t2", "t1"]}
    )
    reference = pl.DataFrame(
        {"token": ["The", "the", "dog"], "file_id": ["r1", "r2", "r1"]}
    )
    result = keywords(target, reference, term, "ll", min_target_range=1)

    name = term.meta.output_name()
    assert "target_range" in result.columns
    got = dict(zip(result[name], result["target_range"]))
    assert got == {"cat": 2, "dog": 1}


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_indirect_term(method: str) -> None:
    # An expression need not name its column outright: the schema settles it.
    expected = keywords(TARGET, REFERENCE, "norm", method)
    got = keywords(TARGET, REFERENCE, pl.col("^nor.*$"), method)
    assert_frame_equal(expected, got, check_row_order=False)


def test_keywords_multi_column_term() -> None:
    with pytest.raises(ValueError, match="expr must identify a single column"):
        keywords(TARGET, REFERENCE, pl.col("norm", "file_id"), "ll")


@pytest.mark.parametrize(
    "threshold,value,expected",
    [
        # Thresholds are inclusive lower bounds (>=).
        ("min_target_freq", 0, {"cat", "dog", "fish", "the"}),
        ("min_target_freq", 2, {"cat", "dog", "the"}),  # fish (freq=1) excluded
        ("min_target_freq", 3, {"cat", "the"}),  # dog (freq=2) also excluded
        ("min_target_range", 0, {"cat", "dog", "fish", "the"}),
        ("min_target_range", 2, {"cat", "the"}),  # dog/fish (range=1) excluded
        ("min_target_range", 3, {"the"}),  # only "the" reaches range=3
    ],
)
def test_keywords_frequency_thresholds(
    threshold: str, value: int, expected: set[str]
) -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ll", **{threshold: value})
    assert set(result["norm"]) == expected


@pytest.mark.parametrize(
    "k,match",
    [
        (None, "needs a value for k"),
        (0, "positive"),
        (-1, "positive"),
    ],
)
def test_keywords_smp_k(k: int | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        keywords(TARGET, REFERENCE, pl.col("norm"), "smp", k=k)


def test_keywords_k_ignored_by_other_methods() -> None:
    with pytest.warns(UserWarning, match="only used when method='smp'"):
        keywords(TARGET, REFERENCE, pl.col("norm"), "ll", k=1)


@pytest.mark.parametrize(
    "threshold,value,expected",
    [
        # ttest reports only the words overrepresented in the target, so "the"
        # is absent throughout; the thresholds cut that set down further.
        ("min_target_freq", 0, {"cat", "dog", "fish"}),
        ("min_target_freq", 2, {"cat", "dog"}),  # fish (freq=1) excluded
        ("min_target_freq", 3, {"cat"}),  # dog (freq=2) also excluded
        ("min_target_range", 2, {"cat"}),  # dog/fish (range=1) excluded
        ("min_target_range", 3, set()),  # only "the" reaches range=3, and it is not key
    ],
)
def test_keywords_ttest_frequency_thresholds(
    threshold: str, value: int, expected: set[str]
) -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest", **{threshold: value})
    assert set(result["norm"]) == expected


def test_keywords_ttest_thresholds_keep_statistics() -> None:
    # The thresholds pick which words are reported; they must not disturb the
    # per-file relative frequencies the test is computed from, whose
    # denominator is every token in the file.
    full = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest")
    filtered = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest", min_target_freq=2)
    assert_frame_equal(
        full.filter(pl.col("norm").is_in(filtered["norm"].implode())), filtered
    )


def test_keywords_ttest() -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest")

    assert result.columns == [
        "norm",
        "target_freq",
        "target_range",
        "t",
        "p",
        "df",
        "g",
    ]
    # The target counts the thresholds cut on are reported alongside the test.
    got = dict(zip(result["norm"], result["target_range"]))
    assert got == {word: TARGET_RANGE[word] for word in result["norm"]}
    # ttest only reports words overrepresented in the target (t > 0).
    assert (result["t"] > 0).all()
    # Hedges' g shares the t-statistic's sign.
    assert (result["g"] > 0).all()
    # Ranked by p-value, ascending.
    pvals = result["p"].to_list()
    assert pvals == sorted(pvals)


def test_keywords_ttest_keeps_target_only_words() -> None:
    """A word absent from the reference is a keyword, not a missing sample.

    It occurs in none of the reference files, so its relative frequency there
    is zero in every one of them -- a constant sample, which Welch's test still
    has a standard error for.
    """
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest")

    target_only = set(TARGET["norm"]) - set(REFERENCE["norm"])
    assert target_only <= set(result["norm"])


@pytest.mark.parametrize("method", ["ll", "pmi", "chisq", "ttest"])
def test_keywords_lazy_matches_eager(method: str) -> None:
    eager = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    lazy = keywords(TARGET.lazy(), REFERENCE.lazy(), pl.col("norm"), method).collect()
    # Tie-break order between engines isn't guaranteed (e.g. words with equal PMI).
    assert_frame_equal(eager, lazy, check_row_order=False)


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_file_id_column(method: str) -> None:
    # A corpus that names its file column something other than "file_id" must
    # give the same answer once `file_id_column` points at it.
    renamed_target = TARGET.rename({"file_id": "text_id"})
    renamed_reference = REFERENCE.rename({"file_id": "text_id"})
    expected = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    got = keywords(
        renamed_target,
        renamed_reference,
        pl.col("norm"),
        method,
        file_id_column="text_id",
    )
    assert_frame_equal(expected, got, check_row_order=False)


NULL_ROWS = pl.DataFrame({"norm": [None, None], "file_id": ["t1", "t2"]})


@pytest.mark.parametrize("method", ["ll", "pmi", "chisq", "minsens", "ttest"])
@pytest.mark.parametrize("part", ["target", "reference"])
def test_keywords_drops_null_terms(method: str, part: str) -> None:
    # A null term is not an occurrence of anything, so it gets no row of its own
    # -- and the corpus totals the measures divide by count only what survives.
    frames = {"target": TARGET, "reference": REFERENCE}
    expected = keywords(frames["target"], frames["reference"], "norm", method)
    frames[part] = pl.concat([frames[part], NULL_ROWS])
    got = keywords(frames["target"], frames["reference"], "norm", method)
    assert None not in got["norm"].to_list()
    assert_frame_equal(expected, got, check_row_order=False)


@pytest.mark.parametrize("method", ["ll", "pmi", "chisq", "minsens", "ttest"])
@pytest.mark.parametrize("part", ["target", "reference"])
def test_keywords_drops_null_file_ids(method: str, part: str) -> None:
    # A token with no file id is in no document, so it must not add a document
    # of its own to the counts `min_target_range` and 'ttest' work from.
    frames = {"target": TARGET, "reference": REFERENCE}
    dropped = {
        part: frames[part].filter(pl.col("file_id") != frames[part]["file_id"][0])
    }
    expected = keywords(
        dropped.get("target", TARGET),
        dropped.get("reference", REFERENCE),
        "norm",
        method,
    )
    frames[part] = frames[part].with_columns(
        pl.when(pl.col("file_id") == frames[part]["file_id"][0])
        .then(None)
        .otherwise(pl.col("file_id"))
        .alias("file_id")
    )
    got = keywords(frames["target"], frames["reference"], "norm", method)
    assert_frame_equal(expected, got, check_row_order=False)


def test_keywords_null_file_id_not_a_document() -> None:
    # "the" is in all three target files; blanking one file id must leave it a
    # range of 2, not 3 with null counted as a file of its own.
    target = TARGET.with_columns(
        pl.when(pl.col("file_id") == "t3")
        .then(None)
        .otherwise(pl.col("file_id"))
        .alias("file_id")
    )
    result = keywords(target, REFERENCE, "norm", "ll")
    assert dict(zip(result["norm"], result["target_range"])) == {
        "cat": 2,
        "dog": 1,
        "the": 2,
    }


def test_keywords_lazy_drops_nulls() -> None:
    # The lazy path drops on the same terms as the eager one.
    target = pl.concat([TARGET, NULL_ROWS]).lazy()
    got = keywords(target, REFERENCE.lazy(), "norm", "ll").collect()
    assert None not in got["norm"].to_list()


# --- Measures of the caller's own ---------------------------------------------

# The built-ins that are plain functions of the counts. 'smp' takes an extra k
# and 'ttest' is not a function of the counts at all, so both are out.
BUILTIN_FUNCTIONS = [
    ("pmi", "PMI", pmi),
    ("mi3", "MI3", mi3),
    ("ll", "LogLik", loglik),
    ("bic", "BIC", bic),
    ("chisq", "ChiSq", chisq),
    ("tscore", "TScore", tscore),
    ("zscore", "ZScore", zscore),
    ("minsens", "MinSens", minsens),
    ("logratio", "LogRatio", logratio),
    ("pctdiff", "%DIFF", pctdiff),
    ("oddsratio", "OddsRatio", oddsratio),
]


@pytest.mark.parametrize(
    "method,col,function", BUILTIN_FUNCTIONS, ids=[m for m, _, _ in BUILTIN_FUNCTIONS]
)
def test_keywords_builtin_function_matches_its_name(
    method: str, col: str, function
) -> None:
    """Passing a measure's function ranks exactly as naming it does."""
    by_name = keywords(TARGET, REFERENCE, "norm", method)
    by_function = keywords(TARGET, REFERENCE, "norm", function)

    assert by_function.columns[-1] == function.__name__
    assert_frame_equal(
        by_function.rename({function.__name__: col}), by_name, check_row_order=False
    )


@pytest.mark.parametrize(
    "measure,column",
    [(jaccard, "jaccard"), (named_by_alias, "Jaccard")],
    ids=["named by def", "named by alias"],
)
def test_keywords_own_measure(measure, column: str) -> None:
    result = keywords(TARGET, REFERENCE, "norm", measure)

    assert result.columns == ["norm", "freqs", "target_range", column]
    # Ranked strongest first, as the built-in measures are.
    assert result[column].to_list() == sorted(result[column], reverse=True)
    fields = [pl.col("freqs").struct.field(name) for name in ("f12", "f1", "f2", "n")]
    expected = result.select(measure(*fields)).to_series()
    assert result[column].to_list() == pytest.approx(expected.to_list())


def test_keywords_own_measure_still_warns_about_k() -> None:
    with pytest.warns(UserWarning, match="only used when method='smp'"):
        keywords(TARGET, REFERENCE, "norm", jaccard, k=1)


@pytest.mark.parametrize(
    "measure,message",
    [
        (lambda f12, f1, f2, n: f12 / f1, "needs a name for the column"),
        (lambda f12, f1, f2, n: 3, "must return a polars expression"),
    ],
    ids=["unnamed", "not an expression"],
)
def test_keywords_bad_own_measure(measure, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        keywords(TARGET, REFERENCE, "norm", measure)


def test_keywords_invalid_method() -> None:
    # check_choice's own behaviour is covered in test_utils; here, that the
    # message offers every method keywords() actually implements.
    with pytest.raises(
        ValueError,
        match=(
            "ttest, pmi, mi3, ll, bic, chisq, tscore, zscore, minsens, smp, "
            "logratio, pctdiff, oddsratio"
        ),
    ):
        keywords(TARGET, REFERENCE, pl.col("norm"), "bogus")


def test_keywords_names_the_corpus_that_was_wrong() -> None:
    # Two frames go in, so the message has to say which one to look at.
    with pytest.raises(ValueError, match="the target corpus must be a polars"):
        keywords("corpus", REFERENCE, pl.col("norm"), "ll")
    with pytest.raises(ValueError, match="the reference corpus must be a polars"):
        keywords(TARGET, "corpus", pl.col("norm"), "ll")


@pytest.mark.parametrize("empty", ["target", "reference"])
def test_keywords_empty_corpus(empty: str) -> None:
    # An empty reference would otherwise give every word a keyness of 0.
    frames = {"target": TARGET, "reference": REFERENCE}
    frames[empty] = frames[empty].clear()
    with pytest.raises(ValueError, match=f"the {empty} corpus is empty"):
        keywords(frames["target"], frames["reference"], pl.col("norm"), "ll")


@pytest.mark.parametrize("method", ["ll", "ttest"])
@pytest.mark.parametrize("missing", ["target", "reference"])
def test_keywords_missing_term_column(method: str, missing: str) -> None:
    # A typo, or two corpora that name the same annotation differently.
    frames = {"target": TARGET, "reference": REFERENCE}
    frames[missing] = frames[missing].rename({"norm": "lemma"})
    with pytest.raises(ValueError, match=f"the {missing} corpus has no column 'norm'"):
        keywords(frames["target"], frames["reference"], "norm", method)


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_missing_file_id_column(method: str) -> None:
    with pytest.raises(ValueError, match="Use file_id_column= to point at"):
        keywords(TARGET.drop("file_id"), REFERENCE, pl.col("norm"), method)


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_ignores_other_columns(method: str) -> None:
    # Corpora annotated differently still compare: only the columns `expr` and
    # `file_id_column` name are read, so the schemas need not match.
    target = TARGET.with_columns(pos=pl.lit("N"))
    reference = REFERENCE.with_columns(genre=pl.lit("news"), c5=pl.lit("NN1"))
    expected = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    got = keywords(target, reference, pl.col("norm"), method)
    assert_frame_equal(expected, got, check_row_order=False)
