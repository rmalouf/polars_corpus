import polars as pl
import pyparsing as pp
import pytest

from polars_corpus.search import Span
from polars_corpus.cqp import matchall


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
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(3, 4)

    def test_single_word_negative_match(self, sample_corpus):
        """Test inequality constrain on a single specific word"""
        query = '[word!="fox"]'
        matches = matchall(sample_corpus, query)
        assert len(matches) == 8
        assert Span(3, 4) not in matches

    def test_pos_tag_match(self, sample_corpus):
        """Test matching by part-of-speech tag"""
        query = '[pos="JJ"]'
        matches = matchall(sample_corpus, query)
        # Should match "quick", "brown", "lazy"
        assert len(matches) == 3
        assert Span(1, 2) in matches
        assert Span(2, 3) in matches
        assert Span(7, 8) in matches

    def test_lemma_match(self, sample_corpus):
        """Test matching by lemma"""
        query = '[lemma="the"]'
        matches = matchall(sample_corpus, query)
        # Should match both "The" and "the"
        assert len(matches) == 2
        assert Span(0, 1) in matches
        assert Span(6, 7) in matches

    def test_no_match(self, sample_corpus):
        """Test query that should return no matches"""
        query = '[word="elephant"]'
        matches = matchall(sample_corpus, query)
        assert matches is None


class TestSequenceMatching:
    """Test matching sequences of tokens"""

    def test_two_token_sequence(self, sample_corpus):
        """Test matching a sequence of two tokens"""
        query = '[pos="JJ"] [pos="NN"]'
        matches = matchall(sample_corpus, query)
        # Should match "brown fox" and "lazy dog"
        assert len(matches) == 2
        assert Span(2, 4) in matches
        assert Span(7, 9) in matches

    def test_three_token_sequence(self, sample_corpus):
        """Test matching a sequence of three tokens"""
        query = '[pos="DT"] [pos="JJ"] [pos="NN"]'
        matches = matchall(sample_corpus, query)
        # Should match "the lazy dog"
        assert len(matches) == 1
        assert matches[0] == Span(6, 9)

    def test_specific_word_sequence(self, sample_corpus):
        """Test matching specific word sequences"""
        query = '[word="the"] [word="lazy"]'
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(6, 8)


class TestConstraintOperators:
    """Test logical operators in constraints"""

    def test_conjunction_constraint(self, sample_corpus):
        """Test AND constraint within a token"""
        query = '[pos="JJ" & lemma="brown"]'
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(2, 3)

    def test_disjunction_constraint(self, sample_corpus):
        """Test OR constraint within a token"""
        query = '[pos="DT" | pos="NN"]'
        matches = matchall(sample_corpus, query)
        # Should match "The", "fox", "the", "dog"
        assert len(matches) == 4
        assert Span(0, 1) in matches
        assert Span(3, 4) in matches
        assert Span(6, 7) in matches
        assert Span(8, 9) in matches

    def test_complex_constraint(self, sample_corpus):
        """Test complex constraint with multiple operators"""
        query = '[pos="JJ" & (lemma="quick" | lemma="lazy")]'
        matches = matchall(sample_corpus, query)
        # Should match "quick" and "lazy"
        assert len(matches) == 2
        assert Span(1, 2) in matches
        assert Span(7, 8) in matches


class TestWildcardMatching:
    """Test wildcard/skip matching"""

    def test_empty_token(self, sample_corpus):
        """Test matching any token with []"""
        query = "[]"
        matches = matchall(sample_corpus, query)
        # Should match every single token
        assert len(matches) == len(sample_corpus)
        for i in range(len(sample_corpus)):
            assert Span(i, i + 1) in matches

    def test_wildcard_in_sequence(self, sample_corpus):
        """Test wildcard within a sequence"""
        query = '[pos="DT"] [] [pos="NN"]'
        matches = matchall(sample_corpus, query)
        # Should match "the lazy dog"
        assert len(matches) == 1
        assert matches[0] == Span(6, 9)


class TestQuantifiers:
    """Test quantifier operators (*, +, ?, and numeric bounds)"""

    def test_zero_or_more_quantifier(self, complex_corpus):
        """Test * quantifier (zero or more)"""
        query = '[pos="JJ"]* [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "long winding paved street", "red barn", "cow"
        assert len(matches) == 4
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(10, 14)
        assert matches[2] == Span(15, 17)
        assert matches[3] == Span(18, 19)

    def test_one_or_more_quantifier(self, complex_corpus):
        """Test + quantifier (one or more)"""
        query = '[pos="JJ"]+ [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "long winding paved street", "red barn"
        assert len(matches) == 3
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(10, 14)
        assert matches[2] == Span(15, 17)

    def test_optional_quantifier(self, sample_corpus):
        """Test ? quantifier (zero or one)"""
        query = '[pos="DT"]? [pos="JJ"] [pos="NN"]'
        matches = matchall(sample_corpus, query)
        # Should match "brown fox", "the lazy dog"
        assert len(matches) == 2
        assert matches[0] == Span(2, 4)
        assert matches[1] == Span(6, 9)

    def test_exact_count_quantifier(self, complex_corpus):
        """Test {n} quantifier (exactly n occurrences)"""
        query = '[pos="JJ"]{2} [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "winding paved street"
        assert len(matches) == 2
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(11, 14)

    def test_range_quantifier(self, complex_corpus):
        """Test {m,n} quantifier (between m and n occurrences)"""
        query = '[pos="JJ"]{1,2} [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "winding paved street", "red barn"
        assert len(matches) == 3
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(11, 14)
        assert matches[2] == Span(15, 17)

    def test_min_quantifier(self, complex_corpus):
        """Test {m,} quantifier (at least m occurrences)"""
        query = '[pos="JJ"]{2,} [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "long winding paved street"
        assert len(matches) == 2
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(10, 14)

    def test_max_quantifier(self, complex_corpus):
        """Test {,n} quantifier (at most n occurrences)"""
        query = '[pos="JJ"]{,2} [pos="NN"]'
        matches = matchall(complex_corpus, query)
        # Should match "big red house", "winding paved street", "red barn", "cow"
        assert len(matches) == 4
        assert matches[0] == Span(5, 8)
        assert matches[1] == Span(11, 14)
        assert matches[2] == Span(15, 17)
        assert matches[3] == Span(18, 19)


class TestDisjunction:
    """Test disjunction at the pattern level"""

    def test_pattern_disjunction(self, sample_corpus):
        """Test OR between different patterns"""
        query = '[pos="DT"] | [pos="VBZ"]'
        matches = matchall(sample_corpus, query)
        # Should match "The", "the", "jumps"
        assert len(matches) == 3
        assert matches[0] == Span(0, 1)
        assert matches[1] == Span(4, 5)
        assert matches[2] == Span(6, 7)

    def test_complex_pattern_disjunction(self, sample_corpus):
        """Test OR between complex patterns"""
        query = '[pos="JJ"] [pos="NN"] | [pos="DT"] [pos="JJ"]'
        matches = matchall(sample_corpus, query)
        # Should match "The quick", "brown fox", "the lazy"
        assert len(matches) == 3
        assert matches[0] == Span(0, 2)
        assert matches[1] == Span(2, 4)
        assert matches[2] == Span(6, 8)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_corpus(self):
        """Test matching on empty corpus"""
        empty_corpus = pl.DataFrame({"word": [], "pos": [], "lemma": []})
        query = '[pos="NN"]'
        matches = matchall(empty_corpus, query)
        assert matches is None

    def test_single_token_corpus(self):
        """Test matching on single token corpus"""
        single_corpus = pl.DataFrame(
            {"word": ["test"], "pos": ["NN"], "lemma": ["test"]}
        )
        query = '[pos="NN"]'
        matches = matchall(single_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(0, 1)

    def test_pattern_longer_than_corpus(self, sample_corpus):
        """Test pattern that's longer than the corpus"""
        # Create a very long pattern
        long_pattern = " ".join(['[pos="NN"]'] * 20)
        matches = matchall(sample_corpus, long_pattern)
        assert matches is None


class TestRegexPatterns:
    """Test regex-style pattern matching in constraints"""

    def test_case_sensitive_matching(self, sample_corpus):
        """Test case sensitive word matching"""
        query = '[word="THE"]'  # Uppercase
        matches = matchall(sample_corpus, query)
        # Should not match since "THE" != "The"
        assert matches is None

        query = '[word="The"]'  # Correct case
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(0, 1)

    def test_regex_wildcard_patterns(self, sample_corpus):
        """Test regex wildcard patterns"""
        query = '[word=".*ox"]'  # Ends with "ox"
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(3, 4)  # "fox"

    def test_regex_character_classes(self, sample_corpus):
        """Test regex character classes"""
        query = '[word="[Tt]he"]'  # "The" or "the"
        matches = matchall(sample_corpus, query)
        assert len(matches) == 2
        assert Span(0, 1) in matches  # "The"
        assert Span(6, 7) in matches  # "the"

    def test_regex_alternation(self, sample_corpus):
        """Test regex alternation patterns"""
        query = '[word="quick|brown"]'  # "quick" or "brown"
        matches = matchall(sample_corpus, query)
        assert len(matches) == 2
        assert Span(1, 2) in matches  # "quick"
        assert Span(2, 3) in matches  # "brown"

    def test_regex_pos_patterns(self, sample_corpus):
        """Test regex patterns on POS tags"""
        query = '[pos="[JN].*"]'  # Starts with J or N
        matches = matchall(sample_corpus, query)
        # Should match JJ (adjectives) and NN (nouns)
        assert len(matches) == 5  # quick, brown, fox, lazy, dog

    def test_regex_anchors(self, sample_corpus):
        """Test regex anchors (^ and $)"""
        query = '[lemma="^the$"]'  # Exact match for "the"
        matches = matchall(sample_corpus, query)
        assert len(matches) == 2

    def test_regex_quantifiers_in_patterns(self, sample_corpus):
        """Test regex quantifiers within value patterns"""
        query = '[word="do.?"]'  # "do" followed by optional character
        matches = matchall(sample_corpus, query)
        assert len(matches) == 1
        assert matches[0] == Span(8, 9)  # "dog"


@pytest.mark.parametrize(
    "query,expected_count",
    [
        ('[pos="JJ"]', 3),  # All adjectives
        ('[pos="NN"]', 2),  # All nouns
        ('[pos="DT"]', 2),  # All determiners
        ('[word="the"]', 1),  # Lowercase "the" only
        ('[pos="JJ"] [pos="NN"]', 2),  # Adjective-noun sequences
    ],
)
def test_parametrized_queries(sample_corpus, query, expected_count):
    """Parametrized tests for various query patterns"""
    matches = matchall(sample_corpus, query)
    assert len(matches) == expected_count


class TestPerformance:
    """Test performance-related aspects"""

    def test_large_corpus_handling(self):
        """Test handling of larger corpora (basic performance test)"""
        # Create a moderately large corpus
        large_corpus = pl.DataFrame(
            {"word": ["test"] * 1000, "pos": ["NN"] * 1000, "lemma": ["test"] * 1000}
        )

        query = '[pos="NN"]'
        matches = matchall(large_corpus, query)
        assert len(matches) == 1000

    def test_complex_query_performance(self, sample_corpus):
        """Test performance with complex queries"""
        complex_query = '([pos="DT"] [pos="JJ"]* [pos="NN"]) | ([pos="VBZ"] []?)'
        matches = matchall(sample_corpus, complex_query)
        # Should complete without timeout/error
        assert isinstance(matches, list)


class TestErrorHandling:
    """Test error conditions and malformed queries"""

    def test_malformed_bracket_syntax(self, sample_corpus):
        """Test malformed bracket syntax"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"'))  # Missing closing bracket

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, 'pos="NN"]'))  # Missing opening bracket

    def test_invalid_feature_names(self, sample_corpus):
        """Test queries with non-existent feature names"""
        with pytest.raises((ValueError, KeyError, pl.exceptions.ColumnNotFoundError)):
            list(matchall(sample_corpus, '[invalid_feature="value"]'))

    def test_malformed_regex_patterns(self, sample_corpus):
        """Test malformed regex patterns in values"""
        with pytest.raises((ValueError, Exception)):  # Regex compilation error
            list(matchall(sample_corpus, '[word="[unclosed"]'))

        with pytest.raises((ValueError, Exception)):
            list(matchall(sample_corpus, '[word="*invalid"]'))  # Invalid regex

    def test_unmatched_parentheses(self, sample_corpus):
        """Test unmatched parentheses in queries"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '([pos="NN"]'))  # Missing closing paren

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"])'))  # Missing opening paren

    def test_invalid_quantifier_syntax(self, sample_corpus):
        """Test invalid quantifier syntax"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"]{'))  # Incomplete quantifier

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"]{-1}'))  # Negative quantifier

        with pytest.raises((ValueError, pp.ParseException)):
            list(
                matchall(sample_corpus, '[pos="NN"]{2,1}')
            )  # Invalid range (max < min)

    def test_malformed_constraint_logic(self, sample_corpus):
        """Test malformed logical operators in constraints"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN" &]'))  # Dangling operator

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[& pos="NN"]'))  # Leading operator

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN" pos="VB"]'))  # Missing operator

    def test_empty_query(self, sample_corpus):
        """Test empty query string"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, ""))

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, "   "))  # Whitespace only

    def test_incomplete_disjunction(self, sample_corpus):
        """Test incomplete disjunction patterns"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"] |'))  # Dangling OR

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '| [pos="NN"]'))  # Leading OR

    def test_nested_quantifiers(self, sample_corpus):
        """Test invalid nested quantifiers"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"]*+'))  # Multiple quantifiers

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos="NN"]?*'))  # Multiple quantifiers

    def test_invalid_constraint_syntax(self, sample_corpus):
        """Test invalid constraint syntax"""
        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, "[pos=]"))  # Missing value

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[="NN"]'))  # Missing feature

        with pytest.raises((ValueError, pp.ParseException)):
            list(matchall(sample_corpus, '[pos"NN"]'))  # Missing equals

    def test_unicode_handling_errors(self, sample_corpus):
        """Test potential unicode/encoding errors"""
        # Test with various problematic unicode characters
        test_cases = [
            '[word="\x00"]',  # Null character
            '[word="\uffff"]',  # High unicode
        ]

        for query in test_cases:
            try:
                result = matchall(sample_corpus, query)
                # If it doesn't raise an error, result should be empty
                assert result is None
            except (ValueError, UnicodeError, pp.ParseException):
                # These exceptions are acceptable for problematic unicode
                pass

    def test_extremely_long_query(self, sample_corpus):
        """Test handling of extremely long queries"""
        # Create a very long but valid query
        long_pattern = " ".join(['[pos="NN"]'] * 1000)
        try:
            matches = matchall(sample_corpus, long_pattern)
            assert matches is None  # Should be no matches for this long pattern
        except (MemoryError, RecursionError):
            # These are acceptable for extremely long queries
            pytest.skip("System limits reached for extremely long query")

    def test_deeply_nested_patterns(self, sample_corpus):
        """Test deeply nested parenthetical patterns"""
        # Create deeply nested pattern
        nested = '[pos="NN"]'
        for _ in range(100):
            nested = f"({nested})"

        try:
            matches = matchall(sample_corpus, nested)
            assert isinstance(matches, list)
        except (RecursionError, ValueError):
            # Acceptable for deeply nested patterns
            pytest.skip("System limits reached for deeply nested pattern")


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


if __name__ == "__main__":
    pytest.main([__file__])
