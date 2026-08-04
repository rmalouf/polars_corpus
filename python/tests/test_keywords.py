import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_corpus import keywords

# Target corpus. Per-word term frequency (tf) and document frequency (df):
#   cat : tf=3, df=2 (t1, t2)
#   dog : tf=2, df=1 (t1)
#   fish: tf=1, df=1 (t3)
#   the : tf=3, df=3 (t1, t2, t3)
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

# Expected target document frequencies keyed by word.
TARGET_DF = {"cat": 2, "dog": 1, "fish": 1, "the": 3}


@pytest.mark.parametrize(
    "method,col,kwargs",
    [
        ("ll", "LogLik", {}),
        ("pmi", "PMI", {}),
        ("chisq", "ChiSq", {}),
        ("smp", "SMP", {"k": 1}),
    ],
)
def test_keywords_assoc_structure(method: str, col: str, kwargs: dict) -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), method, **kwargs)

    assert result.columns == ["norm", "freqs", "target_df", col]
    # Only target-corpus words appear, and the part marker is dropped.
    assert set(result["norm"]) == {"cat", "dog", "fish", "the"}
    # Ranked by association strength, descending.
    vals = result[col].to_list()
    assert vals == sorted(vals, reverse=True)


def test_keywords_target_df_values() -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ll")
    got = dict(zip(result["norm"], result["target_df"]))
    assert got == TARGET_DF


@pytest.mark.parametrize(
    "term",
    [
        pl.col("token").str.to_lowercase().alias("norm"),  # aliased
        pl.col("token").str.to_lowercase(),  # keeps root name "token"
    ],
)
def test_keywords_computed_term(term: pl.Expr) -> None:
    # A computed term is never materialized on `combined`; group_by evaluates it,
    # and its output name must line up with the crosstab column for target_df.
    target = pl.DataFrame(
        {"token": ["Cat", "CAT", "dog"], "file_id": ["t1", "t2", "t1"]}
    )
    reference = pl.DataFrame(
        {"token": ["The", "the", "dog"], "file_id": ["r1", "r2", "r1"]}
    )
    result = keywords(target, reference, term, "ll", min_target_df=1)

    name = term.meta.output_name()
    assert "target_df" in result.columns
    got = dict(zip(result[name], result["target_df"]))
    assert got == {"cat": 2, "dog": 1}


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_indirect_term(method: str) -> None:
    # An expression need not name its column outright: the schema settles it.
    expected = keywords(TARGET, REFERENCE, "norm", method)
    got = keywords(TARGET, REFERENCE, pl.col("^nor.*$"), method)
    assert_frame_equal(expected, got, check_row_order=False)


def test_keywords_term_naming_different_columns() -> None:
    # A pattern that matches a differently named column in each corpus would
    # otherwise fail as a schema mismatch when the two are concatenated.
    reference = REFERENCE.rename({"norm": "normalized"})
    with pytest.raises(ValueError, match="expr names a different column"):
        keywords(TARGET, reference, pl.col("^nor.*$"), "ll")


def test_keywords_multi_column_term() -> None:
    with pytest.raises(ValueError, match="expr must identify a single column"):
        keywords(TARGET, REFERENCE, pl.col("norm", "file_id"), "ll")


@pytest.mark.parametrize("method", ["ll", "pmi", "ttest"])
def test_keywords_string_term(method: str) -> None:
    # A bare column name must work on every method, including ttest, and match
    # passing the equivalent pl.col() expression.
    from_str = keywords(TARGET, REFERENCE, "norm", method)
    from_expr = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    assert from_str.columns[0] == "norm"
    assert_frame_equal(from_str, from_expr, check_row_order=False)


@pytest.mark.parametrize(
    "threshold,value,expected",
    [
        # Thresholds are inclusive lower bounds (>=).
        ("min_target_tf", 0, {"cat", "dog", "fish", "the"}),
        ("min_target_tf", 2, {"cat", "dog", "the"}),  # fish (tf=1) excluded
        ("min_target_tf", 3, {"cat", "the"}),  # dog (tf=2) also excluded
        ("min_target_df", 0, {"cat", "dog", "fish", "the"}),
        ("min_target_df", 2, {"cat", "the"}),  # dog/fish (df=1) excluded
        ("min_target_df", 3, {"the"}),  # only "the" reaches df=3
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


@pytest.mark.parametrize("threshold", ["min_target_tf", "min_target_df"])
def test_keywords_ttest_ignores_thresholds(threshold: str) -> None:
    # ttest can't apply them, so say so rather than silently returning everything.
    with pytest.warns(UserWarning, match="not applied when method='ttest'"):
        keywords(TARGET, REFERENCE, pl.col("norm"), "ttest", **{threshold: 2})


def test_keywords_ttest() -> None:
    result = keywords(TARGET, REFERENCE, pl.col("norm"), "ttest")

    assert result.columns == ["norm", "stat", "pval", "df"]
    # ttest only reports words overrepresented in the target (stat > 0).
    assert (result["stat"] > 0).all()
    # Ranked by p-value, ascending.
    pvals = result["pval"].to_list()
    assert pvals == sorted(pvals)


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


def test_keywords_invalid_method() -> None:
    # check_choice's own behaviour is covered in test_utils; here, that the
    # message offers every method keywords() actually implements.
    with pytest.raises(ValueError, match="ttest, pmi, ll, chisq, smp, minsens"):
        keywords(TARGET, REFERENCE, pl.col("norm"), "bogus")


@pytest.mark.parametrize("method", ["LL", " ll ", "TTest"])
def test_keywords_method_case_insensitive(method: str) -> None:
    got = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    expected = keywords(TARGET, REFERENCE, pl.col("norm"), method.strip().lower())
    assert_frame_equal(got, expected)


@pytest.mark.parametrize("bad", [[1, 2, 3], "corpus", None, TARGET["norm"]])
def test_keywords_invalid_corpus(bad: object) -> None:
    with pytest.raises(ValueError, match="the target corpus must be a polars"):
        keywords(bad, REFERENCE, pl.col("norm"), "ll")
    with pytest.raises(ValueError, match="the reference corpus must be a polars"):
        keywords(TARGET, bad, pl.col("norm"), "ll")


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


def test_keywords_invalid_term() -> None:
    with pytest.raises(ValueError, match="not a Series"):
        keywords(TARGET, REFERENCE, TARGET["norm"], "ll")
    with pytest.raises(ValueError, match="got int"):
        keywords(TARGET, REFERENCE, 3, "ll")


@pytest.mark.parametrize("method", ["ll", "ttest"])
def test_keywords_ignores_other_columns(method: str) -> None:
    # Corpora annotated differently still compare: only the columns `expr` and
    # `file_id_column` name are read, so the schemas need not match.
    target = TARGET.with_columns(pos=pl.lit("N"))
    reference = REFERENCE.with_columns(genre=pl.lit("news"), c5=pl.lit("NN1"))
    expected = keywords(TARGET, REFERENCE, pl.col("norm"), method)
    got = keywords(target, reference, pl.col("norm"), method)
    assert_frame_equal(expected, got, check_row_order=False)
