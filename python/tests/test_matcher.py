import polars as pl
import pyparsing as pp
import pytest
from polars_corpus.matcher import Match, Span, get_matches


def match_spans(matches):
    """Extract spans from matches for easier testing."""
    return [m.span for m in matches]


def match_bindings(matches, var_name):
    """Extract spans for a specific variable from matches."""
    return [m.bindings.get(var_name) for m in matches if var_name in m.bindings]


@pytest.fixture
def sample_corpus():
    """Sample corpus with typical linguistic annotations"""
    return pl.DataFrame(
        {
            "word": [
                "The",
                "quick",
                "brown",
                "fox",
                "jumps",
                "over",
                "the",
                "lazy",
                "dog",
            ],
            "pos": ["DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN"],
            "lemma": [
                "the",
                "quick",
                "brown",
                "fox",
                "jump",
                "over",
                "the",
                "lazy",
                "dog",
            ],
        }
    )


@pytest.fixture
def complex_corpus():
    """More complex corpus for advanced pattern testing"""
    return pl.DataFrame(
        {
            "word": [
                "John",
                "walked",
                "slowly",
                "to",
                "the",
                "big",
                "red",
                "house",
                "yesterday",
                "the",
                "long",
                "winding",
                "paved",
                "street",
                "the",
                "red",
                "barn",
                "the",
                "cow",
            ],
            "pos": [
                "NNP",
                "VBD",
                "RB",
                "TO",
                "DT",
                "JJ",
                "JJ",
                "NN",
                "RB",
                "DT",
                "JJ",
                "JJ",
                "JJ",
                "NN",
                "DT",
                "JJ",
                "NN",
                "DT",
                "NN",
            ],
            "lemma": [
                "john",
                "walk",
                "slowly",
                "to",
                "the",
                "big",
                "red",
                "house",
                "yesterday",
                "the",
                "long",
                "winding",
                "paved",
                "street",
                "the",
                "red",
                "barn",
                "the",
                "cow",
            ],
        }
    )


class TestBasicTokenMatching:
    """Test basic token-level matching functionality"""

    def test_single_word_match(self, sample_corpus):
        """Test equality constraint on a single specific word"""
        query = '[word="fox"]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(3, 4)

    def test_single_word_negative_match(self, sample_corpus):
        """Test inequality constrain on a single specific word"""
        query = '[word!="fox"]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 8
        assert Span(3, 4) not in match_spans(matches)

    def test_pos_tag_match(self, sample_corpus):
        """Test matching by part-of-speech tag"""
        query = '[pos="JJ"]'
        matches = get_matches(sample_corpus, query)
        # Should match "quick", "brown", "lazy"
        assert len(matches) == 3
        assert Span(1, 2) in match_spans(matches)
        assert Span(2, 3) in match_spans(matches)
        assert Span(7, 8) in match_spans(matches)

    def test_lemma_match(self, sample_corpus):
        """Test matching by lemma"""
        query = '[lemma="the"]'
        matches = get_matches(sample_corpus, query)
        # Should match both "The" and "the"
        assert len(matches) == 2
        assert Span(0, 1) in match_spans(matches)
        assert Span(6, 7) in match_spans(matches)

    def test_case_insensitive_matching(self, sample_corpus):
        """Test case-insensitive matching with %c modifier"""
        # Basic case-insensitive match
        query = '[word="the"%c]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 2
        assert Span(0, 1) in match_spans(matches)  # "The"
        assert Span(6, 7) in match_spans(matches)  # "the"

        # Case-insensitive inequality
        query = '[word!="THE"%c]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 7  # Everything except "The" and "the"

        # With boolean expressions
        query = '[pos="JJ" & word="BROWN"%c]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(2, 3)  # "brown"

    def test_no_match(self, sample_corpus):
        """Test query that should return no matches"""
        query = '[word="elephant"]'
        matches = get_matches(sample_corpus, query)
        assert matches is None


class TestSequenceMatching:
    """Test matching sequences of tokens"""

    def test_two_token_sequence(self, sample_corpus):
        """Test matching a sequence of two tokens"""
        query = '[pos="JJ"] [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        # Should match "brown fox" and "lazy dog"
        assert len(matches) == 2
        assert Span(2, 4) in match_spans(matches)
        assert Span(7, 9) in match_spans(matches)

    def test_three_token_sequence(self, sample_corpus):
        """Test matching a sequence of three tokens"""
        query = '[pos="DT"] [pos="JJ"] [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        # Should match "the lazy dog"
        assert len(matches) == 1
        assert matches[0].span == Span(6, 9)

    def test_specific_word_sequence(self, sample_corpus):
        """Test matching specific word sequences"""
        query = '[word="the"] [word="lazy"]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(6, 8)


class TestConstraintOperators:
    """Test logical operators in constraints"""

    def test_conjunction_constraint(self, sample_corpus):
        """Test AND constraint within a token"""
        query = '[pos="JJ" & lemma="brown"]'
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(2, 3)

    def test_disjunction_constraint(self, sample_corpus):
        """Test OR constraint within a token"""
        query = '[pos="DT" | pos="NN"]'
        matches = get_matches(sample_corpus, query)
        # Should match "The", "fox", "the", "dog"
        assert len(matches) == 4
        assert Span(0, 1) in match_spans(matches)
        assert Span(3, 4) in match_spans(matches)
        assert Span(6, 7) in match_spans(matches)
        assert Span(8, 9) in match_spans(matches)

    def test_complex_constraint(self, sample_corpus):
        """Test complex constraint with multiple operators"""
        query = '[pos="JJ" & (lemma="quick" | lemma="lazy")]'
        matches = get_matches(sample_corpus, query)
        # Should match "quick" and "lazy"
        assert len(matches) == 2
        assert Span(1, 2) in match_spans(matches)
        assert Span(7, 8) in match_spans(matches)


class TestWildcardMatching:
    """Test wildcard/skip matching"""

    def test_empty_token(self, sample_corpus):
        """Test matching any token with []"""
        query = "[]"
        matches = get_matches(sample_corpus, query)
        # Should match every single token
        assert len(matches) == len(sample_corpus)
        for i in range(len(sample_corpus)):
            assert Span(i, i + 1) in match_spans(matches)

    def test_wildcard_in_sequence(self, sample_corpus):
        """Test wildcard within a sequence"""
        query = '[pos="DT"] [] [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        # Should match "the lazy dog"
        assert len(matches) == 1
        assert matches[0].span == Span(6, 9)


class TestQuantifiers:
    """Test quantifier operators (*, +, ?, and numeric bounds)"""

    def test_zero_or_more_quantifier(self, complex_corpus):
        """Test * quantifier (zero or more)"""
        query = '[pos="JJ"]* [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "long winding paved street", "red barn", "cow"
        assert len(matches) == 4
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(10, 14)
        assert matches[2].span == Span(15, 17)
        assert matches[3].span == Span(18, 19)

    def test_one_or_more_quantifier(self, complex_corpus):
        """Test + quantifier (one or more)"""
        query = '[pos="JJ"]+ [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "long winding paved street", "red barn"
        assert len(matches) == 3
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(10, 14)
        assert matches[2].span == Span(15, 17)

    def test_optional_quantifier(self, sample_corpus):
        """Test ? quantifier (zero or one)"""
        query = '[pos="DT"]? [pos="JJ"] [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        # Should match "brown fox", "the lazy dog"
        assert len(matches) == 2
        assert matches[0].span == Span(2, 4)
        assert matches[1].span == Span(6, 9)

    def test_exact_count_quantifier(self, complex_corpus):
        """Test {n} quantifier (exactly n occurrences)"""
        query = '[pos="JJ"]{2} [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "winding paved street"
        assert len(matches) == 2
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(11, 14)

    def test_range_quantifier(self, complex_corpus):
        """Test {m,n} quantifier (between m and n occurrences)"""
        query = '[pos="JJ"]{1,2} [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "winding paved street", "red barn"
        assert len(matches) == 3
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(11, 14)
        assert matches[2].span == Span(15, 17)

    def test_min_quantifier(self, complex_corpus):
        """Test {m,} quantifier (at least m occurrences)"""
        query = '[pos="JJ"]{2,} [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "long winding paved street"
        assert len(matches) == 2
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(10, 14)

    def test_max_quantifier(self, complex_corpus):
        """Test {,n} quantifier (at most n occurrences)"""
        query = '[pos="JJ"]{,2} [pos="NN"]'
        matches = get_matches(complex_corpus, query)
        # Should match "big red house", "winding paved street", "red barn", "cow"
        assert len(matches) == 4
        assert matches[0].span == Span(5, 8)
        assert matches[1].span == Span(11, 14)
        assert matches[2].span == Span(15, 17)
        assert matches[3].span == Span(18, 19)


class TestDisjunction:
    """Test disjunction at the pattern level"""

    def test_pattern_disjunction(self, sample_corpus):
        """Test OR between different patterns"""
        query = '[pos="DT"] | [pos="VBZ"]'
        matches = get_matches(sample_corpus, query)
        # Should match "The", "the", "jumps"
        assert len(matches) == 3
        assert matches[0].span == Span(0, 1)
        assert matches[1].span == Span(4, 5)
        assert matches[2].span == Span(6, 7)

    def test_complex_pattern_disjunction(self, sample_corpus):
        """Test OR between complex patterns"""
        query = '[pos="JJ"] [pos="NN"] | [pos="DT"] [pos="JJ"]'
        matches = get_matches(sample_corpus, query)
        # Should match "The quick", "brown fox", "the lazy"
        assert len(matches) == 3
        assert matches[0].span == Span(0, 2)
        assert matches[1].span == Span(2, 4)
        assert matches[2].span == Span(6, 8)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_corpus(self):
        """Test matching on empty corpus"""
        empty_corpus = pl.DataFrame({"word": [], "pos": [], "lemma": []})
        query = '[pos="NN"]'
        matches = get_matches(empty_corpus, query)
        assert matches is None

    def test_single_token_corpus(self):
        """Test matching on single token corpus"""
        single_corpus = pl.DataFrame(
            {"word": ["test"], "pos": ["NN"], "lemma": ["test"]}
        )
        query = '[pos="NN"]'
        matches = get_matches(single_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(0, 1)

    def test_pattern_longer_than_corpus(self, sample_corpus):
        """Test pattern that's longer than the corpus"""
        # Create a very long pattern
        long_pattern = " ".join(['[pos="NN"]'] * 20)
        matches = get_matches(sample_corpus, long_pattern)
        assert matches is None


class TestRegexPatterns:
    """Test regex-style pattern matching in constraints"""

    def test_regex_patterns(self, sample_corpus):
        """Test various regex patterns"""
        # Wildcard patterns
        query = '[word=".*ox"]'  # Ends with "ox"
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(3, 4)  # "fox"

        # Character classes
        query = '[word="[Tt]he"]'  # "The" or "the"
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 2
        assert Span(0, 1) in match_spans(matches)  # "The"
        assert Span(6, 7) in match_spans(matches)  # "the"

        # Alternation
        query = '[word="quick|brown"]'  # "quick" or "brown"
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 2
        assert Span(1, 2) in match_spans(matches)  # "quick"
        assert Span(2, 3) in match_spans(matches)  # "brown"

        # POS patterns
        query = '[pos="[JN].*"]'  # Starts with J or N
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 5  # quick, brown, fox, lazy, dog

        # Anchors
        query = '[lemma="^the$"]'  # Exact match for "the"
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 2
        assert Span(0, 1) in match_spans(matches)
        assert Span(6, 7) in match_spans(matches)

        # Quantifiers in patterns
        query = '[word="do.?"]'  # "do" followed by optional character
        matches = get_matches(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0].span == Span(8, 9)  # "dog"


# @pytest.mark.parametrize(
#     "query,expected_count",
#     [
#         ('[pos="JJ"]', 3),  # All adjectives
#         ('[pos="NN"]', 2),  # All nouns
#         ('[pos="DT"]', 2),  # All determiners
#         ('[word="the"]', 1),  # Lowercase "the" only
#         ('[pos="JJ"] [pos="NN"]', 2),  # Adjective-noun sequences
#     ],
# )
# def test_parametrized_queries(sample_corpus, query, expected_count):
#     """Parametrized tests for various query patterns"""
#     matches = get_matches(sample_corpus, query)
#     assert len(matches) == expected_count


# class TestPerformance:
#     """Test performance-related aspects"""
#
#     def test_large_corpus_handling(self):
#         """Test handling of larger corpora (basic performance test)"""
#         # Create a moderately large corpus
#         large_corpus = pl.DataFrame(
#             {"word": ["test"] * 1000, "pos": ["NN"] * 1000, "lemma": ["test"] * 1000}
#         )
#
#         query = '[pos="NN"]'
#         matches = get_matches(large_corpus, query)
#         assert len(matches) == 1000
#
#     def test_complex_query_performance(self, sample_corpus):
#         """Test performance with complex queries"""
#         complex_query = '([pos="DT"] [pos="JJ"]* [pos="NN"]) | ([pos="VBZ"] []?)'
#         matches = get_matches(sample_corpus, complex_query)
#         # Should complete without timeout/error
#         assert isinstance(matches, list)


class TestErrorHandling:
    """Test error conditions and malformed queries"""

    def test_malformed_syntax(self, sample_corpus):
        """Test various malformed syntax errors"""
        error_cases = [
            '[pos="NN"',  # Missing closing bracket
            'pos="NN"]',  # Missing opening bracket
            '([pos="NN"]',  # Missing closing paren
            '[pos="NN"])',  # Missing opening paren
            '[pos="NN"]{',  # Incomplete quantifier
            '[pos="NN" &]',  # Dangling operator
            '[pos="NN"] |',  # Dangling OR
            "[pos=]",  # Missing value
            "",  # Empty query
        ]

        for query in error_cases:
            with pytest.raises((ValueError, pp.ParseException)):
                list(get_matches(sample_corpus, query))

    def test_invalid_features_and_regex(self, sample_corpus):
        """Test invalid feature names and regex patterns"""
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            list(get_matches(sample_corpus, '[invalid_feature="value"]'))

        with pytest.raises((ValueError, Exception)):
            list(get_matches(sample_corpus, '[word="[unclosed"]'))


class TestSpanUtilities:
    """Test span-related utility functions"""

    def test_span_creation(self):
        """Test Span namedtuple creation"""
        span = Span(1, 5)
        assert span.start == 1
        assert span.end == 5
        assert len(span) == 2

    def test_span_equality(self):
        """Test Span equality comparison"""
        span1 = Span(1, 5)
        span2 = Span(1, 5)
        span3 = Span(1, 6)

        assert span1 == span2
        assert span1 != span3

    def test_span_in_list(self):
        """Test Span membership in lists"""
        spans = [Span(1, 3), Span(5, 7), Span(10, 12)]
        assert Span(1, 3) in spans
        assert Span(1, 4) not in spans


class TestVariableBindings:
    """Test CQP variable binding functionality ($var: pattern syntax)"""

    @pytest.mark.parametrize(
        "query,var,expected_span,description",
        [
            ('$n: [pos="NN"]', "n", Span(3, 4), "single token"),
            (
                '$det: [pos="DT"] $adj: [pos="JJ"] $noun: [pos="NN"]',
                "det",
                Span(6, 7),
                "multiple vars - first",
            ),
            (
                '[pos="DT"] $adj: [pos="JJ"] [pos="NN"]',
                "adj",
                Span(7, 8),
                "mixed bound/unbound",
            ),
        ],
    )
    def test_basic_bindings(self, sample_corpus, query, var, expected_span, description):
        """Test basic variable binding patterns"""
        matches = get_matches(sample_corpus, query)
        assert matches is not None, f"No matches for: {description}"
        assert len(matches) > 0, f"Empty matches for: {description}"
        assert var in matches[0].bindings, f"Variable '{var}' not in bindings for: {description}"
        assert matches[0].bindings[var] == expected_span, f"Wrong span for {description}"

    def test_multiple_variables_in_sequence(self, sample_corpus):
        """Test that multiple variables are all captured simultaneously"""
        query = '$det: [pos="DT"] $adj: [pos="JJ"] $noun: [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        assert matches is not None
        assert len(matches) == 1

        # Verify all three variables are captured
        assert "det" in matches[0].bindings
        assert "adj" in matches[0].bindings
        assert "noun" in matches[0].bindings

        # Verify their spans
        assert matches[0].bindings["det"] == Span(6, 7)  # "the"
        assert matches[0].bindings["adj"] == Span(7, 8)  # "lazy"
        assert matches[0].bindings["noun"] == Span(8, 9)  # "dog"

        # Verify overall match span
        assert matches[0].span == Span(6, 9)

    @pytest.mark.parametrize(
        "query,var,expected_match_idx,expected_span,description",
        [
            (
                '$adjs: [pos="JJ"]+ [pos="NN"]',
                "adjs",
                0,
                Span(5, 7),
                "plus: all consecutive adjectives",
            ),
            (
                '$adjs: [pos="JJ"]* [pos="NN"]',
                "adjs",
                -1,
                Span(18, 18),
                "star: zero-match empty span",
            ),
            (
                '$det: [pos="DT"]? [pos="JJ"] [pos="NN"]',
                "det",
                1,
                Span(6, 7),
                "optional: present",
            ),
            (
                '$det: [pos="DT"]? [pos="JJ"] [pos="NN"]',
                "det",
                0,
                Span(2, 2),
                "optional: absent empty span",
            ),
            (
                '$two: [pos="JJ"]{2} [pos="NN"]',
                "two",
                0,
                Span(5, 7),
                "exact count: 2 adjectives",
            ),
        ],
    )
    def test_quantifier_bindings(
        self, complex_corpus, query, var, expected_match_idx, expected_span, description
    ):
        """Test that quantifiers bind entire matched sequence, not just last token"""
        matches = get_matches(complex_corpus, query)
        assert matches is not None, f"No matches for: {description}"
        assert len(matches) > 0, f"Empty matches for: {description}"

        match = matches[expected_match_idx]
        assert var in match.bindings, f"Variable '{var}' not in bindings for: {description}"
        assert match.bindings[var] == expected_span, f"Wrong span for {description}"

    def test_nested_bindings(self, sample_corpus):
        """Test that nested variable bindings capture both outer and inner variables"""
        query = '$phrase: ($det: [pos="DT"]) [pos="JJ"] [pos="NN"]'
        matches = get_matches(sample_corpus, query)
        assert matches is not None
        assert len(matches) == 1

        # Both variables should be captured
        assert "phrase" in matches[0].bindings
        assert "det" in matches[0].bindings

        # Verify spans
        assert matches[0].bindings["det"] == Span(6, 7)  # "the"
        assert matches[0].bindings["phrase"] == Span(6, 9)  # "the lazy dog"

        # Overall match should equal phrase binding
        assert matches[0].span == matches[0].bindings["phrase"]

    def test_binding_in_alternation(self, sample_corpus):
        """Test that variable bindings work correctly with alternation (disjunction)"""
        query = '$target: ([pos="JJ"] | [pos="NN"])'
        matches = get_matches(sample_corpus, query)
        assert matches is not None

        # Should match all JJ and NN: quick, brown, fox, lazy, dog
        assert len(matches) == 5

        # Each match should have the "target" binding
        for match in matches:
            assert "target" in match.bindings
            # Binding should match the overall span for single-variable queries
            assert match.bindings["target"] == match.span

    @pytest.mark.parametrize(
        "query",
        [
            '$x: [pos="JJ"] $x: [pos="NN"]',  # Sequential reuse
            '$x: [pos="JJ"]+ $x: [pos="NN"]+',  # With quantifiers
            '($x: [pos="JJ"]) ($x: [pos="NN"])',  # In groups
        ],
    )
    def test_variable_reuse_error(self, sample_corpus, query):
        """Variable names cannot be reused in same query - should raise error"""
        with pytest.raises((ValueError, RuntimeError, pp.ParseException)):
            get_matches(sample_corpus, query)


if __name__ == "__main__":
    pytest.main([__file__])
