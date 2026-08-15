import polars as pl
import pytest
from lark.exceptions import LarkError
from polars_corpus.matcher import Span, run_query, search, search_cqp

from .helpers import corpus, spans


@pytest.fixture
def sample_corpus():
    """Sample corpus with typical linguistic annotations"""
    return corpus(
        word="The quick brown fox jumps over the lazy dog",
        pos="DT JJ JJ NN VBZ IN DT JJ NN",
        lemma="the quick brown fox jump over the lazy dog",
    )


@pytest.fixture
def complex_corpus():
    """More complex corpus for advanced pattern testing"""
    return corpus(
        word="John walked slowly to the big red house yesterday "
        "the long winding paved street the red barn the cow",
        pos="NNP VBD RB TO DT JJ JJ NN RB DT JJ JJ JJ NN DT JJ NN DT NN",
        lemma="john walk slowly to the big red house yesterday "
        "the long winding paved street the red barn the cow",
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        # --- single-token constraints ---
        ('[word="fox"]', [(3, 4)]),
        (
            '[word!="fox"]',
            [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)],
        ),
        ('[pos="JJ"]', [(1, 2), (2, 3), (7, 8)]),
        ('[lemma="the"]', [(0, 1), (6, 7)]),
        ('[word="elephant"]', []),
        # --- case folding with %c ---
        ('[word="the"%c]', [(0, 1), (6, 7)]),
        (
            '[word!="THE"%c]',
            [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8), (8, 9)],
        ),
        ('[pos="JJ" & word="BROWN"%c]', [(2, 3)]),
        # --- boolean operators within a token ---
        ('[pos="JJ" & lemma="brown"]', [(2, 3)]),
        ('[pos="DT" | pos="NN"]', [(0, 1), (3, 4), (6, 7), (8, 9)]),
        ('[pos="JJ" & (lemma="quick" | lemma="lazy")]', [(1, 2), (7, 8)]),
        # --- sequences ---
        ('[pos="JJ"] [pos="NN"]', [(2, 4), (7, 9)]),
        ('[pos="DT"] [pos="JJ"] [pos="NN"]', [(6, 9)]),
        ('[word="the"] [word="lazy"]', [(6, 8)]),
        # --- wildcards ---
        ("[]", [(i, i + 1) for i in range(9)]),
        ('[pos="DT"] [] [pos="NN"]', [(6, 9)]),
        ('[pos="DT"]? [pos="JJ"] [pos="NN"]', [(2, 4), (6, 9)]),
        # --- pattern-level disjunction ---
        ('[pos="DT"] | [pos="VBZ"]', [(0, 1), (4, 5), (6, 7)]),
        (
            '[pos="JJ"] [pos="NN"] | [pos="DT"] [pos="JJ"]',
            [(0, 2), (2, 4), (6, 8)],
        ),
        # --- regex in constraint values ---
        ('[word=".*ox"]', [(3, 4)]),
        ('[word="[Tt]he"]', [(0, 1), (6, 7)]),
        ('[word="quick|brown"]', [(1, 2), (2, 3)]),
        ('[pos="[JN].*"]', [(1, 2), (2, 3), (3, 4), (7, 8), (8, 9)]),
        ('[lemma="^the$"]', [(0, 1), (6, 7)]),
        ('[word="do.?"]', [(8, 9)]),
        # --- a pattern longer than the corpus cannot match ---
        (" ".join(['[pos="NN"]'] * 20), []),
    ],
)
def test_queries(sample_corpus, query, expected):
    matches, _ = run_query(sample_corpus, query)
    assert spans(matches) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ('[pos="JJ"]* [pos="NN"]', [(5, 8), (10, 14), (15, 17), (18, 19)]),
        ('[pos="JJ"]+ [pos="NN"]', [(5, 8), (10, 14), (15, 17)]),
        ('[pos="JJ"]{2} [pos="NN"]', [(5, 8), (11, 14)]),
        ('[pos="JJ"]{1,2} [pos="NN"]', [(5, 8), (11, 14), (15, 17)]),
        ('[pos="JJ"]{2,} [pos="NN"]', [(5, 8), (10, 14)]),
        ('[pos="JJ"]{,2} [pos="NN"]', [(5, 8), (11, 14), (15, 17), (18, 19)]),
    ],
)
def test_quantifiers(complex_corpus, query, expected):
    matches, _ = run_query(complex_corpus, query)
    assert spans(matches) == expected


@pytest.mark.parametrize(
    "corpus_columns,expected",
    [
        ({"word": [], "pos": [], "lemma": []}, []),
        ({"word": ["test"], "pos": ["NN"], "lemma": ["test"]}, [(0, 1)]),
    ],
    ids=["empty", "single-token"],
)
def test_degenerate_corpora(corpus_columns, expected):
    df = pl.DataFrame(corpus_columns)
    matches, _ = run_query(df, '[pos="NN"]')
    assert spans(matches) == expected


@pytest.mark.parametrize(
    "query",
    [
        '[pos="NN"',  # Missing closing bracket
        'pos="NN"]',  # Missing opening bracket
        '([pos="NN"]',  # Missing closing paren
        '[pos="NN"])',  # Missing opening paren
        '[pos="NN"]{',  # Incomplete quantifier
        '[pos="NN" &]',  # Dangling operator
        '[pos="NN"] |',  # Dangling OR
        "[pos=]",  # Missing value
        "",  # Empty query
    ],
)
def test_malformed_syntax(sample_corpus, query):
    with pytest.raises((ValueError, LarkError)):
        run_query(sample_corpus, query)


def test_unknown_feature(sample_corpus):
    with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
        run_query(sample_corpus, '[invalid_feature="value"]')


def test_invalid_regex(sample_corpus):
    """A well-formed query whose constraint value is not a valid regex"""
    with pytest.raises(pl.exceptions.ComputeError):
        run_query(sample_corpus, '[word="[unclosed"]')


@pytest.mark.parametrize("fn,query", [(search, "fox"), (search_cqp, '[word="fox"]')])
def test_lazy_corpus_rejected(sample_corpus, fn, query):
    """Matching walks the corpus by position, so a LazyFrame cannot stand in."""
    with pytest.raises(ValueError, match="must be an eager polars DataFrame"):
        fn(sample_corpus.lazy(), query)


class TestSpan:
    """Span is implemented in Rust, so its dunders need exercising"""

    def test_fields_and_length(self):
        span = Span(1, 5)
        assert (span.start, span.end) == (1, 5)
        assert len(span) == 2
        assert (span[0], span[1]) == (1, 5)

    def test_equality(self):
        assert Span(1, 5) == Span(1, 5)
        assert Span(1, 5) != Span(1, 6)

    def test_membership(self):
        spans_ = [Span(1, 3), Span(5, 7)]
        assert Span(1, 3) in spans_
        assert Span(1, 4) not in spans_


class TestVariableBindings:
    """CQP variable binding functionality ($var: pattern syntax)"""

    @pytest.mark.parametrize(
        "query,var,expected_span",
        [
            pytest.param('$n: ([pos="NN"])', "n", Span(3, 4), id="single-token"),
            pytest.param(
                '$det: ([pos="DT"]) $adj: ([pos="JJ"]) $noun: ([pos="NN"])',
                "det",
                Span(6, 7),
                id="multiple-vars",
            ),
            pytest.param(
                '[pos="DT"] $adj: ([pos="JJ"]) [pos="NN"]',
                "adj",
                Span(7, 8),
                id="mixed-bound-unbound",
            ),
            pytest.param(
                '$v: ([pos="JJ"] | [pos="NN"])',
                "v",
                Span(1, 2),
                id="disjunction-first-alternative",
            ),
            pytest.param(
                '$v: ([pos="VB"] | [pos="NN"])',
                "v",
                Span(3, 4),
                id="disjunction-backtracking",
            ),
        ],
    )
    def test_basic_bindings(self, sample_corpus, query, var, expected_span):
        matches, _ = run_query(sample_corpus, query)
        assert matches
        assert matches[0].bindings[var] == expected_span

    def test_multiple_variables_captured_simultaneously(self, sample_corpus):
        query = '$det: ([pos="DT"]) $adj: ([pos="JJ"]) $noun: ([pos="NN"])'
        matches, _ = run_query(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].bindings == {
            "det": Span(6, 7),  # "the"
            "adj": Span(7, 8),  # "lazy"
            "noun": Span(8, 9),  # "dog"
        }
        assert matches[0].span == Span(6, 9)

    @pytest.mark.parametrize(
        "query,expected",
        [
            pytest.param('[pos="NN"]', [], id="no-bindings"),
            pytest.param(
                '$det: ([pos="DT"]) $adj: ([pos="JJ"]) $noun: ([pos="NN"])',
                ["det", "adj", "noun"],
                id="binding-order",
            ),
            pytest.param(
                '$phrase: (($det: ([pos="DT"])) [pos="JJ"] [pos="NN"])',
                ["det", "phrase"],
                id="inner-binding-closes-first",
            ),
            pytest.param(
                '($vb: ([pos="VB"]) | [pos="NN"])',
                ["vb"],
                id="named-though-no-match-bound-it",
            ),
        ],
    )
    def test_variables_reported(self, sample_corpus, query, expected):
        """The names come off the query, in the order it closes them"""
        _, variables = run_query(sample_corpus, query)

        assert variables == expected

    @pytest.mark.parametrize(
        "query,var,match_idx,expected_span",
        [
            pytest.param(
                '$adjs: ([pos="JJ"]+) [pos="NN"]',
                "adjs",
                0,
                Span(5, 7),
                id="plus-all-consecutive",
            ),
            pytest.param(
                '$adjs: ([pos="JJ"]*) [pos="NN"]',
                "adjs",
                -1,
                Span(18, 18),
                id="star-zero-match-empty-span",
            ),
            pytest.param(
                '$det: ([pos="DT"]?) [pos="JJ"] [pos="NN"]',
                "det",
                2,
                Span(14, 15),
                id="optional-present",
            ),
            pytest.param(
                '$det: ([pos="DT"]?) [pos="JJ"] [pos="NN"]',
                "det",
                0,
                Span(6, 6),
                id="optional-absent-empty-span",
            ),
            pytest.param(
                '$two: ([pos="JJ"]{2}) [pos="NN"]',
                "two",
                0,
                Span(5, 7),
                id="exact-count",
            ),
        ],
    )
    def test_quantifier_bindings(
        self, complex_corpus, query, var, match_idx, expected_span
    ):
        """Quantifiers bind the entire matched sequence, not just the last token"""
        matches, _ = run_query(complex_corpus, query)
        assert matches
        assert matches[match_idx].bindings[var] == expected_span

    def test_nested_bindings(self, sample_corpus):
        """Nested bindings capture both the outer and the inner variable"""
        query = '$phrase: (($det: ([pos="DT"])) [pos="JJ"] [pos="NN"])'
        matches, _ = run_query(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].bindings == {
            "det": Span(6, 7),  # "the"
            "phrase": Span(6, 9),  # "the lazy dog"
        }
        assert matches[0].span == matches[0].bindings["phrase"]

    def test_binding_in_alternation(self, sample_corpus):
        """A single binding around an alternation spans the whole match"""
        matches, _ = run_query(sample_corpus, '$target: ([pos="JJ"] | [pos="NN"])')
        # quick, brown, fox, lazy, dog
        assert len(matches) == 5
        assert all(m.bindings["target"] == m.span for m in matches)

    @pytest.mark.parametrize(
        "query",
        [
            '$x: ([pos="JJ"]) $x: ([pos="NN"])',  # Sequential reuse
            '$x: ([pos="JJ"]+) $x: ([pos="NN"]+)',  # With quantifiers
            '($x: ([pos="JJ"])) ($x: ([pos="NN"]))',  # In groups
        ],
    )
    def test_variable_reuse_error(self, sample_corpus, query):
        """Variable names cannot be reused within one query"""
        with pytest.raises((ValueError, RuntimeError)):
            run_query(sample_corpus, query)


@pytest.fixture
def two_file_corpus():
    """Corpus where 'brown fox' occurs across a file boundary and within a file"""
    return corpus(
        word="the quick brown fox brown fox jumps",
        file_id="d1 d1 d1 d2 d2 d2 d2",
    )


class TestFileBoundaries:
    """Matches must not span a change in the file_id column"""

    @pytest.mark.parametrize(
        "query,file_id_column,expected",
        [
            pytest.param(
                '[word="brown"] [word="fox"]',
                None,
                [(2, 4), (4, 6)],
                id="unrestricted-by-default",
            ),
            pytest.param(
                '[word="brown"] [word="fox"]',
                "file_id",
                [(4, 6)],
                id="straddling-match-suppressed",
            ),
            pytest.param(
                '[word="fox"] [word="brown"]',
                "file_id",
                [(3, 5)],
                id="may-start-on-file-initial-token",
            ),
            pytest.param(
                '[word="quick"] [] [word="fox"]',
                "file_id",
                [],
                id="gap-cannot-cross-boundary",
            ),
            pytest.param(
                '[word="fox"] [] [word="fox"]',
                "file_id",
                [(3, 6)],
                id="gap-within-file-ok",
            ),
            pytest.param(
                '[word="brown"] []* [word="jumps"]',
                "file_id",
                [(4, 7)],
                id="quantified-gap",
            ),
        ],
    )
    def test_boundaries_confine_matches(
        self, two_file_corpus, query, file_id_column, expected
    ):
        matches, _ = run_query(two_file_corpus, query, file_id_column)
        assert spans(matches) == expected

    @pytest.mark.parametrize(
        "df",
        [
            pl.DataFrame({"word": ["brown", "fox"], "file_id": ["d1", "d2"]}),
            pl.DataFrame({"word": []}),  # the check precedes the empty-corpus shortcut
        ],
        ids=["populated", "empty"],
    )
    def test_missing_column_raises(self, df):
        """Naming an absent column is an error, not a silent no-op"""
        with pytest.raises(ValueError, match="not_a_column"):
            run_query(df, '[word="brown"] [word="fox"]', "not_a_column")

    @pytest.mark.parametrize(
        "dtype",
        [pl.String, pl.Categorical, pl.Enum(["d1", "d2"]), pl.UInt32],
    )
    def test_file_id_dtypes(self, two_file_corpus, dtype):
        """Boundaries are found regardless of how file_id is stored"""
        cast = pl.col("file_id")
        if dtype == pl.UInt32:
            cast = cast.str.strip_prefix("d")
        df = two_file_corpus.with_columns(cast.cast(dtype))
        matches, _ = run_query(df, '[word="brown"] [word="fox"]', "file_id")
        assert spans(matches) == [(4, 6)]

    @pytest.mark.parametrize(
        "words,file_ids,expected",
        [
            pytest.param(
                "brown fox brown fox",
                [None, None, "d1", "d1"],
                [(0, 2), (2, 4)],
                id="nulls-form-their-own-file",
            ),
            pytest.param(
                "brown fox",
                ["a", "b"],
                [],
                id="every-token-its-own-file",
            ),
            pytest.param(
                "brown fox brown fox brown fox",
                ["d2", "d2", "d1", "d1", "d2", "d2"],
                [(0, 2), (2, 4), (4, 6)],
                id="unsorted-file-ids-need-no-sort",
            ),
        ],
    )
    def test_run_detection(self, words, file_ids, expected):
        df = pl.DataFrame({"word": words.split(), "file_id": file_ids})
        matches, _ = run_query(df, '[word="brown"] [word="fox"]', "file_id")
        assert spans(matches) == expected

    def test_single_token_corpus(self):
        """n < 2 takes the early return in run_starts"""
        df = pl.DataFrame({"word": ["fox"], "file_id": ["d1"]})
        matches, _ = run_query(df, '[word="fox"]', "file_id")
        assert spans(matches) == [(0, 1)]


@pytest.mark.parametrize(
    "query,expected",
    [
        pytest.param("fox", [(2, 3)], id="bare-word-uses-token-column"),
        pytest.param("_JJ", [(1, 2)], id="bare-tag-uses-pos-column"),
        pytest.param("{jump}", [(3, 4)], id="braces-use-lemma-column"),
        pytest.param("quick_JJ", [(1, 2)], id="word-plus-tag-uses-both"),
    ],
)
def test_role_columns_are_settable(query, expected):
    """search() must route each query construct at the named column"""
    renamed = corpus(w="the quick fox jumped", p="DT JJ NN VBD", l="the quick fox jump")
    results = search(renamed, query, token_column="w", pos_column="p", lemma_column="l")
    assert spans(results._matches) == expected
