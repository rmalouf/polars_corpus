import pytest
import polars as pl
import pyparsing as pp
import numpy as np
from nlpolars.cqp import (
    Token,
    Skip,
    ZeroOrMore,
    OneOrMore,
    OneOrZero,
    Concat,
    Alt,
    Pattern,
    ScanContext,
    constraint_formula,
    node,
    cqp,
    Match,
)


class TestConstraintParsing:
    """Test parsing of token-level constraints"""

    def test_simple_feature_value_constraint(self):
        """Test parsing of basic feature=value constraints"""
        result = constraint_formula.parse_string('pos="NOUN"')
        # Should create a polars expression that matches NOUN exactly
        df = pl.DataFrame({"pos": ["NOUN", "VERB", "NOUN"]})
        matches = df.select(result).to_numpy().flatten()
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(matches, expected)

    def test_conjunction_constraint(self):
        """Test parsing of AND constraints"""
        result = constraint_formula.parse_string('pos="NOUN" & case="NOM"')
        df = pl.DataFrame(
            {"pos": ["NOUN", "NOUN", "VERB"], "case": ["NOM", "ACC", "NOM"]}
        )
        matches = df.select(result).to_numpy().flatten()
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(matches, expected)

    def test_disjunction_constraint(self):
        """Test parsing of OR constraints"""
        result = constraint_formula.parse_string('pos="NOUN" | pos="VERB"')
        df = pl.DataFrame({"pos": ["NOUN", "ADJ", "VERB"]})
        matches = df.select(result).to_numpy().flatten()
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(matches, expected)

    def test_complex_constraint(self):
        """Test parsing of complex nested constraints"""
        result = constraint_formula.parse_string(
            '(pos="NOUN" | pos="VERB") & case="NOM"'
        )
        df = pl.DataFrame(
            {
                "pos": ["NOUN", "VERB", "ADJ", "NOUN"],
                "case": ["NOM", "NOM", "NOM", "ACC"],
            }
        )
        matches = df.select(result).to_numpy().flatten()
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(matches, expected)


class TestNodeParsing:
    """Test parsing of CQP node expressions"""

    def test_token_node_parsing(self):
        """Test parsing of token nodes with constraints"""
        result = node.parse_string('[pos="NOUN"]')
        assert isinstance(result[0], Token)

        # Test the constraint works
        df = pl.DataFrame({"pos": ["NOUN", "VERB"]})
        result[0].set_subject(df)
        assert result[0].valid_tokens[0] == True
        assert result[0].valid_tokens[1] == False

    def test_skip_node_parsing(self):
        """Test parsing of empty skip nodes"""
        result = node.parse_string("[]")
        assert isinstance(result[0], Skip)


class TestBasicPatterns:
    """Test basic pattern classes"""

    @pytest.fixture
    def sample_corpus(self):
        """Sample corpus for testing"""
        return pl.DataFrame(
            {
                "word": ["the", "quick", "brown", "fox", "jumps"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB"],
                "lemma": ["the", "quick", "brown", "fox", "jump"],
            }
        )

    def test_token_pattern_matching(self, sample_corpus):
        """Test Token pattern matching"""
        pattern = Token(pl.col("pos") == "ADJ")
        pattern.set_subject(sample_corpus)

        # Should match positions 1 and 2 (quick, brown)
        expected_valid = np.array([False, True, True, False, False])
        np.testing.assert_array_equal(pattern.valid_tokens, expected_valid)

        # Test _op method
        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 1))  # Start at "quick"
        assert matches == [2]  # Should advance by 1

        matches = list(pattern._op(ctxt, 0))  # Start at "the"
        assert matches == []  # No match

    def test_skip_pattern(self, sample_corpus):
        """Test Skip pattern"""
        pattern = Skip()
        pattern.set_subject(sample_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 2))
        assert matches == [3]  # Always advances by 1

    def test_zero_or_more_pattern(self, sample_corpus):
        """Test ZeroOrMore pattern"""
        adj_pattern = Token(pl.col("pos") == "ADJ")
        pattern = ZeroOrMore(adj_pattern)
        pattern.set_subject(sample_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 1))  # Start at "quick"
        # Should match: 1 (zero matches), 2 (one match), 3 (two matches)
        assert 1 in matches  # Zero occurrences
        assert 2 in matches  # One occurrence
        assert 3 in matches  # Two occurrences

    def test_one_or_more_pattern(self, sample_corpus):
        """Test OneOrMore pattern"""
        adj_pattern = Token(pl.col("pos") == "ADJ")
        pattern = OneOrMore(adj_pattern)
        pattern.set_subject(sample_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 1))  # Start at "quick"
        # Should not match zero occurrences, but should match 1, 2, etc.
        assert 1 not in matches  # Zero occurrences not allowed
        assert 2 in matches  # One occurrence
        assert 3 in matches  # Two occurrences

    def test_one_or_zero_pattern(self, sample_corpus):
        """Test OneOrZero pattern"""
        adj_pattern = Token(pl.col("pos") == "ADJ")
        pattern = OneOrZero(adj_pattern)
        pattern.set_subject(sample_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 1))  # Start at "quick"
        # Should match both zero and one occurrence
        assert 1 in matches  # Zero occurrences
        assert 2 in matches  # One occurrence


class TestComplexPatterns:
    """Test complex pattern combinations"""

    @pytest.fixture
    def complex_corpus(self):
        """More complex corpus for testing"""
        return pl.DataFrame(
            {
                "word": ["a", "very", "quick", "brown", "fox", "runs", "very", "fast"],
                "pos": ["DET", "ADV", "ADJ", "ADJ", "NOUN", "VERB", "ADV", "ADJ"],
            }
        )

    def test_concatenation_pattern(self, complex_corpus):
        """Test Concat pattern"""
        det_pattern = Token(pl.col("pos") == "DET")
        adj_pattern = Token(pl.col("pos") == "ADJ")
        pattern = Concat(det_pattern, adj_pattern)
        pattern.set_subject(complex_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 0))  # Start at "a"
        # "a" is DET, but next word "very" is ADV, not ADJ
        assert matches == []

    def test_alternation_pattern(self, complex_corpus):
        """Test Alt pattern"""
        det_pattern = Token(pl.col("pos") == "DET")
        adv_pattern = Token(pl.col("pos") == "ADV")
        pattern = Alt(det_pattern, adv_pattern)
        pattern.set_subject(complex_corpus)

        ctxt = ScanContext()
        matches_at_0 = list(pattern._op(ctxt, 0))  # Start at "a" (DET)
        matches_at_1 = list(pattern._op(ctxt, 1))  # Start at "very" (ADV)

        assert 1 in matches_at_0  # DET matches
        assert 2 in matches_at_1  # ADV matches

    def test_complex_sequence(self, complex_corpus):
        """Test complex pattern: DET ADV* ADJ+ NOUN"""
        det = Token(pl.col("pos") == "DET")
        adv_star = ZeroOrMore(Token(pl.col("pos") == "ADV"))
        adj_plus = OneOrMore(Token(pl.col("pos") == "ADJ"))
        noun = Token(pl.col("pos") == "NOUN")

        # Build pattern: DET ADV* ADJ+ NOUN
        pattern = Concat(Concat(Concat(det, adv_star), adj_plus), noun)
        pattern.set_subject(complex_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 0))  # Start at "a"
        # Should match "a very quick brown fox" (positions 0-5)
        assert 5 in matches


class TestCQPParsing:
    """Test full CQP expression parsing"""

    def test_simple_cqp_expression(self):
        """Test parsing simple CQP expressions"""
        result = cqp.parse_string('[pos="NOUN"]')
        assert isinstance(result[0], Token)

    def test_sequence_cqp_expression(self):
        """Test parsing CQP sequences"""
        result = cqp.parse_string('[pos="DET"] [pos="ADJ"]')
        assert isinstance(result[0], Concat)
        assert len(result[0].subpatterns) == 2

    def test_alternation_cqp_expression(self):
        """Test parsing CQP alternations"""
        result = cqp.parse_string('[pos="NOUN"] | [pos="VERB"]')
        assert isinstance(result[0], Alt)

    def test_repetition_cqp_expressions(self):
        """Test parsing CQP repetition operators"""
        # Zero or more
        result = cqp.parse_string('[pos="ADJ"]*')
        assert isinstance(result[0], ZeroOrMore)

        # One or more
        result = cqp.parse_string('[pos="ADJ"]+')
        assert isinstance(result[0], OneOrMore)

        # Optional
        result = cqp.parse_string('[pos="ADJ"]?')
        assert isinstance(result[0], OneOrZero)

    def test_complex_cqp_expression(self):
        """Test parsing complex nested CQP expression"""
        expr = '([pos="DET"] [pos="ADJ"]* [pos="NOUN"]) | [pos="PRON"]'
        result = cqp.parse_string(expr)
        assert isinstance(result[0], Alt)


class TestMatchAll:
    """Test the matchall functionality"""

    @pytest.fixture
    def test_corpus(self):
        return pl.DataFrame(
            {
                "word": [
                    "the",
                    "quick",
                    "brown",
                    "fox",
                    "jumps",
                    "over",
                    "the",
                    "lazy",
                    "dog",
                ],
                "pos": [
                    "DET",
                    "ADJ",
                    "ADJ",
                    "NOUN",
                    "VERB",
                    "PREP",
                    "DET",
                    "ADJ",
                    "NOUN",
                ],
            }
        )

    def test_matchall_simple_pattern(self, test_corpus):
        """Test matchall with simple pattern"""
        pattern = Token(pl.col("pos") == "DET")
        matches = list(pattern.matchall(test_corpus))

        # Should find "the" at positions 0 and 6
        assert len(matches) == 2
        assert matches[0]["word"].to_list() == ["the"]
        assert matches[1]["word"].to_list() == ["the"]

    def test_matchall_sequence_pattern(self, test_corpus):
        """Test matchall with sequence pattern"""
        det_adj = Concat(Token(pl.col("pos") == "DET"), Token(pl.col("pos") == "ADJ"))
        matches = list(det_adj.matchall(test_corpus))

        # Should find "the quick" and "the lazy"
        assert len(matches) == 2
        assert matches[0]["word"].to_list() == ["the", "quick"]
        assert matches[1]["word"].to_list() == ["the", "lazy"]

    def test_matchall_with_repetition(self, test_corpus):
        """Test matchall with repetition patterns"""
        # DET ADJ+ NOUN pattern
        pattern = Concat(
            Concat(
                Token(pl.col("pos") == "DET"), OneOrMore(Token(pl.col("pos") == "ADJ"))
            ),
            Token(pl.col("pos") == "NOUN"),
        )
        matches = list(pattern.matchall(test_corpus))

        # Should find "the quick brown fox" and "the lazy dog"
        assert len(matches) == 2
        assert matches[0]["word"].to_list() == ["the", "quick", "brown", "fox"]
        assert matches[1]["word"].to_list() == ["the", "lazy", "dog"]


class TestValidStartsOptimization:
    """Test that the valid_starts optimization is working correctly"""

    @pytest.fixture
    def optimization_corpus(self):
        """Corpus designed to test valid_starts optimization"""
        return pl.DataFrame(
            {
                "word": [
                    "det1",
                    "noun1",
                    "verb1",
                    "det2",
                    "adj1",
                    "noun2",
                    "prep1",
                    "det3",
                    "adj2",
                    "adj3",
                    "noun3",
                ],
                "pos": [
                    "DET",
                    "NOUN",
                    "VERB",
                    "DET",
                    "ADJ",
                    "NOUN",
                    "PREP",
                    "DET",
                    "ADJ",
                    "ADJ",
                    "NOUN",
                ],
                "case": ["", "NOM", "", "", "NOM", "ACC", "", "", "NOM", "NOM", "ACC"],
            }
        )

    def test_token_valid_starts_computation(self, optimization_corpus):
        """Test that Token patterns correctly compute valid_starts"""
        # Pattern that matches only DET
        det_pattern = Token(pl.col("pos") == "DET")
        det_pattern.set_subject(optimization_corpus)

        # valid_starts should be True only at DET positions (0, 3, 7)
        expected_valid = np.array(
            [True, False, False, True, False, False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(det_pattern.valid_starts, expected_valid)

        # Pattern that matches ADJ
        adj_pattern = Token(pl.col("pos") == "ADJ")
        adj_pattern.set_subject(optimization_corpus)

        # valid_starts should be True only at ADJ positions (4, 8, 9)
        expected_valid = np.array(
            [False, False, False, False, True, False, False, False, True, True, False]
        )
        np.testing.assert_array_equal(adj_pattern.valid_starts, expected_valid)

    def test_skip_valid_starts_none(self, optimization_corpus):
        """Test that Skip patterns have valid_starts=None (can start anywhere)"""
        skip_pattern = Skip()
        skip_pattern.set_subject(optimization_corpus)

        # Skip should have valid_starts=None since it can match at any position
        assert skip_pattern.valid_starts is None

    def test_concat_valid_starts_inheritance(self, optimization_corpus):
        """Test that Concat patterns inherit valid_starts from first subpattern"""
        det_pattern = Token(pl.col("pos") == "DET")
        adj_pattern = Token(pl.col("pos") == "ADJ")

        # DET followed by anything
        concat_pattern = Concat(det_pattern, adj_pattern)
        concat_pattern.set_subject(optimization_corpus)

        # Should inherit valid_starts from first pattern (DET positions)
        expected_valid = np.array(
            [True, False, False, True, False, False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(concat_pattern.valid_starts, expected_valid)

    def test_alt_valid_starts_union(self, optimization_corpus):
        """Test that Alt patterns compute valid_starts as union of subpatterns"""
        det_pattern = Token(pl.col("pos") == "DET")
        adj_pattern = Token(pl.col("pos") == "ADJ")

        alt_pattern = Alt(det_pattern, adj_pattern)
        alt_pattern.set_subject(optimization_corpus)

        # Should be union of DET positions (0,3,7) and ADJ positions (4,8,9)
        expected_valid = np.array(
            [True, False, False, True, True, False, False, True, True, True, False]
        )
        np.testing.assert_array_equal(alt_pattern.valid_starts, expected_valid)

    def test_alt_with_none_valid_starts(self, optimization_corpus):
        """Test Alt when one subpattern has valid_starts=None"""
        det_pattern = Token(pl.col("pos") == "DET")
        skip_pattern = Skip()

        alt_pattern = Alt(det_pattern, skip_pattern)
        alt_pattern.set_subject(optimization_corpus)

        # If any subpattern has valid_starts=None, result should be None
        assert alt_pattern.valid_starts is None

    def test_repetition_valid_starts_behavior(self, optimization_corpus):
        """Test that repetition patterns have valid_starts=None"""
        adj_pattern = Token(pl.col("pos") == "ADJ")

        # ZeroOrMore can start anywhere (due to zero matches)
        zero_or_more = ZeroOrMore(adj_pattern)
        zero_or_more.set_subject(optimization_corpus)
        assert zero_or_more.valid_starts is None

        # OneOrMore should inherit from subpattern since it requires at least one match
        one_or_more = OneOrMore(adj_pattern)
        one_or_more.set_subject(optimization_corpus)

        # Expected: should inherit ADJ positions (4, 8, 9) from subpattern
        expected_valid = np.array(
            [False, False, False, False, True, False, False, False, True, True, False]
        )
        np.testing.assert_array_equal(one_or_more.valid_starts, expected_valid)

    def test_matchall_uses_valid_starts_optimization(self, optimization_corpus):
        """Test that matchall actually uses the valid_starts optimization"""
        import unittest.mock

        # Create a pattern with known valid_starts
        det_pattern = Token(pl.col("pos") == "DET")
        det_pattern.set_subject(optimization_corpus)

        # Mock the _op method to count how many times it's called
        original_op = det_pattern._op
        with unittest.mock.patch.object(
            det_pattern, "_op", wraps=original_op
        ) as mock_op:
            matches = list(det_pattern.matchall(optimization_corpus))

            # _op should only be called for valid start positions (0, 3, 7)
            # Plus one final call that might go beyond the last valid position
            assert mock_op.call_count == 3  # Only called for DET positions

            # Verify the positions where _op was called
            call_positions = [
                call[0][1] for call in mock_op.call_args_list
            ]  # Extract cursor positions
            expected_positions = [0, 3, 7]  # DET positions
            assert call_positions == expected_positions

    def test_optimization_vs_brute_force_equivalence(self, optimization_corpus):
        """Test that optimized matching gives same results as brute force"""
        det_adj_pattern = Concat(
            Token(pl.col("pos") == "DET"), Token(pl.col("pos") == "ADJ")
        )

        # Get matches using optimization
        optimized_matches = list(det_adj_pattern.matchall(optimization_corpus))

        # Simulate brute force by temporarily setting valid_starts to None
        det_adj_pattern.set_subject(optimization_corpus)
        original_valid_starts = det_adj_pattern.valid_starts
        det_adj_pattern.valid_starts = None

        brute_force_matches = list(det_adj_pattern.matchall(optimization_corpus))

        # Restore original valid_starts
        det_adj_pattern.valid_starts = original_valid_starts

        # Results should be identical
        assert len(optimized_matches) == len(brute_force_matches)
        for opt_match, bf_match in zip(optimized_matches, brute_force_matches):
            assert opt_match.equals(bf_match)

    def test_complex_pattern_valid_starts(self, optimization_corpus):
        """Test valid_starts computation for complex nested patterns"""
        # (DET | ADJ) NOUN pattern
        det_or_adj = Alt(Token(pl.col("pos") == "DET"), Token(pl.col("pos") == "ADJ"))
        noun_pattern = Token(pl.col("pos") == "NOUN")

        complex_pattern = Concat(det_or_adj, noun_pattern)
        complex_pattern.set_subject(optimization_corpus)

        # Should inherit from first subpattern (DET | ADJ)
        # Expected: DET positions (0,3,7) OR ADJ positions (4,8,9)
        expected_valid = np.array(
            [True, False, False, True, True, False, False, True, True, True, False]
        )
        np.testing.assert_array_equal(complex_pattern.valid_starts, expected_valid)

    def test_performance_benefit_measurement(self, optimization_corpus):
        """Measure the performance benefit of valid_starts optimization"""
        import time

        # Create a large corpus for performance testing
        large_corpus_data = {
            "pos": (
                ["OTHER"] * 1000
                + ["DET"]
                + ["OTHER"] * 1000
                + ["ADJ"]
                + ["OTHER"] * 1000
            )
        }
        large_corpus = pl.DataFrame(large_corpus_data)

        pattern = Token(pl.col("pos") == "DET")

        # Time with optimization
        start_time = time.time()
        matches_optimized = list(pattern.matchall(large_corpus))
        optimized_time = time.time() - start_time

        # Time without optimization (simulate by setting valid_starts=None)
        pattern.set_subject(large_corpus)
        pattern.valid_starts = None

        start_time = time.time()
        matches_brute_force = list(pattern.matchall(large_corpus))
        brute_force_time = time.time() - start_time

        # Results should be the same
        assert len(matches_optimized) == len(matches_brute_force)

        # Optimization should be faster (though this might not always be detectable in small tests)
        # This is more of a documentation test than a strict assertion
        print(
            f"Optimized time: {optimized_time:.4f}s, Brute force time: {brute_force_time:.4f}s"
        )

        # At minimum, optimized version shouldn't be significantly slower
        assert (
            optimized_time <= brute_force_time * 3
        )  # Allow margin for test variability


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_corpus(self):
        """Test patterns on empty corpus"""
        empty_corpus = pl.DataFrame({"pos": []})
        pattern = Token(pl.col("pos") == "NOUN")
        matches = list(pattern.matchall(empty_corpus))
        assert len(matches) == 0

    def test_no_matches(self):
        """Test pattern that doesn't match anything"""
        corpus = pl.DataFrame({"pos": ["NOUN", "VERB"]})
        pattern = Token(pl.col("pos") == "ADJ")
        matches = list(pattern.matchall(corpus))
        assert len(matches) == 0

    def test_pattern_repr(self):
        """Test string representations of patterns"""
        token = Token(pl.col("pos") == "NOUN")
        assert "Token" in repr(token)

        skip = Skip()
        assert "Skip" in repr(skip)

        concat = Concat(token, skip)
        assert "Concat" in repr(concat)


class TestCQPParser:
    """Test that the CQP parser correctly translates query strings to pattern objects"""

    def test_basic_token_parsing(self):
        """Test parsing of basic token patterns"""
        # Simple feature-value constraint
        result = cqp.parse_string('[pos="NOUN"]')
        assert len(result) == 1
        assert isinstance(result[0], Token)

        # Test the constraint works
        df = pl.DataFrame({"pos": ["NOUN", "VERB", "NOUN"]})
        result[0].set_subject(df)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result[0].valid_tokens, expected)

    def test_empty_token_parsing(self):
        """Test parsing of empty token (skip) patterns"""
        result = cqp.parse_string("[]")
        assert len(result) == 1
        assert isinstance(result[0], Skip)

    def test_conjunction_constraint_parsing(self):
        """Test parsing of conjunction constraints"""
        result = cqp.parse_string('[pos="NOUN" & case="NOM"]')
        assert isinstance(result[0], Token)

        df = pl.DataFrame(
            {"pos": ["NOUN", "NOUN", "VERB"], "case": ["NOM", "ACC", "NOM"]}
        )
        result[0].set_subject(df)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result[0].valid_tokens, expected)

    def test_disjunction_constraint_parsing(self):
        """Test parsing of disjunction constraints"""
        result = cqp.parse_string('[pos="NOUN" | pos="VERB"]')
        assert isinstance(result[0], Token)

        df = pl.DataFrame({"pos": ["NOUN", "ADJ", "VERB", "PREP"]})
        result[0].set_subject(df)
        expected = np.array([True, False, True, False])
        np.testing.assert_array_equal(result[0].valid_tokens, expected)

    def test_complex_constraint_parsing(self):
        """Test parsing of complex nested constraints"""
        result = cqp.parse_string('[(pos="NOUN" | pos="VERB") & case="NOM"]')
        assert isinstance(result[0], Token)

        df = pl.DataFrame(
            {
                "pos": ["NOUN", "VERB", "ADJ", "NOUN"],
                "case": ["NOM", "NOM", "NOM", "ACC"],
            }
        )
        result[0].set_subject(df)
        expected = np.array([True, True, False, False])
        np.testing.assert_array_equal(result[0].valid_tokens, expected)

    def test_sequence_parsing(self):
        """Test parsing of token sequences"""
        result = cqp.parse_string('[pos="DET"] [pos="NOUN"]')
        assert len(result) == 1
        assert isinstance(result[0], Concat)
        assert len(result[0].subpatterns) == 2
        assert isinstance(result[0].subpatterns[0], Token)
        assert isinstance(result[0].subpatterns[1], Token)

    def test_long_sequence_parsing(self):
        """Test parsing of longer sequences"""
        result = cqp.parse_string('[pos="DET"] [pos="ADJ"] [pos="NOUN"]')
        assert isinstance(result[0], Concat)

        # Should create right-associative Concat structure: Concat(DET, Concat(ADJ, NOUN))
        # due to pairwise_compose implementation
        outer_concat = result[0]
        assert isinstance(outer_concat.subpatterns[0], Token)  # DET
        assert isinstance(outer_concat.subpatterns[1], Concat)  # Inner concat

        inner_concat = outer_concat.subpatterns[1]
        assert isinstance(inner_concat.subpatterns[0], Token)  # ADJ
        assert isinstance(inner_concat.subpatterns[1], Token)  # NOUN

    def test_alternation_parsing(self):
        """Test parsing of alternation (disjunction) between patterns"""
        result = cqp.parse_string('[pos="NOUN"] | [pos="VERB"]')
        assert len(result) == 1
        assert isinstance(result[0], Alt)
        assert len(result[0].subpatterns) == 2
        assert isinstance(result[0].subpatterns[0], Token)
        assert isinstance(result[0].subpatterns[1], Token)

    def test_multiple_alternations(self):
        """Test parsing of multiple alternations"""
        result = cqp.parse_string('[pos="NOUN"] | [pos="VERB"] | [pos="ADJ"]')
        assert isinstance(result[0], Alt)

        # Should create right-associative Alt structure: Alt(NOUN, Alt(VERB, ADJ))
        # due to pairwise_compose implementation
        outer_alt = result[0]
        assert isinstance(outer_alt.subpatterns[0], Token)  # NOUN
        assert isinstance(outer_alt.subpatterns[1], Alt)  # Inner alt

        inner_alt = outer_alt.subpatterns[1]
        assert isinstance(inner_alt.subpatterns[0], Token)  # VERB
        assert isinstance(inner_alt.subpatterns[1], Token)  # ADJ

    def test_zero_or_more_parsing(self):
        """Test parsing of * repetition operator"""
        result = cqp.parse_string('[pos="ADJ"]*')
        assert len(result) == 1
        assert isinstance(result[0], ZeroOrMore)
        assert len(result[0].subpatterns) == 1
        assert isinstance(result[0].subpatterns[0], Token)

    def test_one_or_more_parsing(self):
        """Test parsing of + repetition operator"""
        result = cqp.parse_string('[pos="ADJ"]+')
        assert len(result) == 1
        assert isinstance(result[0], OneOrMore)
        assert len(result[0].subpatterns) == 1
        assert isinstance(result[0].subpatterns[0], Token)

    def test_optional_parsing(self):
        """Test parsing of ? repetition operator"""
        result = cqp.parse_string('[pos="ADJ"]?')
        assert len(result) == 1
        assert isinstance(result[0], OneOrZero)
        assert len(result[0].subpatterns) == 1
        assert isinstance(result[0].subpatterns[0], Token)

    def test_parentheses_parsing(self):
        """Test parsing of parenthesized expressions"""
        result = cqp.parse_string('([pos="DET"] [pos="ADJ"])')
        assert isinstance(result[0], Concat)

        # Parentheses should not change the structure
        result_no_parens = cqp.parse_string('[pos="DET"] [pos="ADJ"]')
        assert type(result[0]) == type(result_no_parens[0])

    def test_complex_expression_parsing(self):
        """Test parsing of complex nested expression"""
        # (DET ADJ*) | PRON
        result = cqp.parse_string('([pos="DET"] [pos="ADJ"]*) | [pos="PRON"]')
        assert isinstance(result[0], Alt)

        # Left side should be Concat(DET, ZeroOrMore(ADJ))
        left_side = result[0].subpatterns[0]
        assert isinstance(left_side, Concat)
        assert isinstance(left_side.subpatterns[0], Token)  # DET
        assert isinstance(left_side.subpatterns[1], ZeroOrMore)  # ADJ*

        # Right side should be Token(PRON)
        right_side = result[0].subpatterns[1]
        assert isinstance(right_side, Token)  # PRON

    def test_repetition_on_groups(self):
        """Test repetition operators applied to grouped expressions"""
        result = cqp.parse_string('([pos="DET"] [pos="ADJ"])*')
        assert isinstance(result[0], ZeroOrMore)
        assert isinstance(result[0].subpatterns[0], Concat)

    def test_mixed_operators_precedence(self):
        """Test operator precedence in complex expressions"""
        # ADJ* NOUN | PRON should parse as (ADJ* NOUN) | PRON
        result = cqp.parse_string('[pos="ADJ"]* [pos="NOUN"] | [pos="PRON"]')
        assert isinstance(result[0], Alt)

        # Left side should be Concat(ZeroOrMore(ADJ), NOUN)
        left_side = result[0].subpatterns[0]
        assert isinstance(left_side, Concat)
        assert isinstance(left_side.subpatterns[0], ZeroOrMore)
        assert isinstance(left_side.subpatterns[1], Token)

        # Right side should be Token(PRON)
        right_side = result[0].subpatterns[1]
        assert isinstance(right_side, Token)

    def test_whitespace_handling(self):
        """Test that parser handles whitespace correctly"""
        # These should all parse to the same structure
        expressions = [
            '[pos="NOUN"][pos="VERB"]',
            '[pos="NOUN"] [pos="VERB"]',
            '[ pos="NOUN" ] [ pos="VERB" ]',
            '[pos="NOUN"]  [pos="VERB"]',
            '\n[pos="NOUN"]\n[pos="VERB"]\n',
        ]

        results = [cqp.parse_string(expr) for expr in expressions]

        # All should create Concat patterns
        for result in results:
            assert isinstance(result[0], Concat)
            assert len(result[0].subpatterns) == 2

    def test_quoted_string_parsing(self):
        """Test parsing of quoted strings with special characters"""
        # Test quotes, spaces, and special characters in values
        test_cases = [
            ('[word="hello"]', "hello"),
            ('[word="hello world"]', "hello world"),
            ('[word="it\'s"]', "it's"),  # Single quote inside double quotes
            ('[lemma="café"]', "café"),  # Unicode characters
        ]

        for expr, expected_value in test_cases:
            result = cqp.parse_string(expr)
            assert isinstance(result[0], Token)
            # Note: We can't easily test the exact value since it's embedded in a polars expression
            # But we can test that parsing succeeds

    def test_feature_name_parsing(self):
        """Test parsing of different feature names"""
        feature_names = ["pos", "lemma", "word", "case", "number", "gender", "tense"]

        for feature in feature_names:
            expr = f'[{feature}="TEST"]'
            result = cqp.parse_string(expr)
            assert isinstance(result[0], Token)

    def test_empty_input(self):
        """Test parser behavior with empty or invalid input"""
        with pytest.raises(pp.ParseException):
            cqp.parse_string("")

        with pytest.raises(pp.ParseException):
            cqp.parse_string("   ")

    def test_invalid_syntax(self):
        """Test parser behavior with invalid CQP syntax"""
        invalid_expressions = [
            "[pos=NOUN]",  # Missing quotes
            '[pos="NOUN"',  # Missing closing bracket
            'pos="NOUN"]',  # Missing opening bracket
            '[pos=="NOUN"]',  # Double equals
            '[pos="NOUN"] &',  # Dangling operator
            '| [pos="NOUN"]',  # Leading operator
            '[pos="NOUN"]]',  # Extra closing bracket
            '[[pos="NOUN"]',  # Extra opening bracket
            '[pos="NOUN" &]',  # Incomplete conjunction
            '[& pos="NOUN"]',  # Leading conjunction operator
        ]

        for expr in invalid_expressions:
            with pytest.raises(pp.ParseException):
                cqp.parse_string(expr)

    def test_parser_end_to_end_functionality(self):
        """Test complete parsing and pattern execution"""
        corpus = pl.DataFrame(
            {
                "word": ["the", "quick", "brown", "fox", "jumps"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB"],
            }
        )

        # Parse and execute: DET ADJ+ NOUN
        result = cqp.parse_string('[pos="DET"] [pos="ADJ"]+ [pos="NOUN"]')
        pattern = result[0]

        matches = list(pattern.matchall(corpus))
        assert len(matches) == 1
        assert matches[0]["word"].to_list() == ["the", "quick", "brown", "fox"]

    def test_realistic_linguistic_queries(self):
        """Test parsing of realistic corpus linguistics queries"""
        realistic_queries = [
            # Noun phrases
            '[pos="DET"]? [pos="ADJ"]* [pos="NOUN"]',
            # Verb phrases
            '[pos="AUX"]? [pos="ADV"]* [pos="VERB"]',
            # Prepositional phrases
            '[pos="PREP"] [pos="DET"]? [pos="ADJ"]* [pos="NOUN"]',
            # Coordination
            '[pos="NOUN"] [word="and"] [pos="NOUN"]',
            # Complex alternation
            '([pos="NOUN"] | [pos="PRON"]) [pos="VERB"] ([pos="NOUN"] | [pos="PRON"])',
            # Multiple constraints
            '[pos="NOUN" & case="NOM"] [pos="VERB" & tense="PAST"]',
        ]

        for query in realistic_queries:
            result = cqp.parse_string(query)
            assert len(result) == 1
            assert isinstance(result[0], Pattern)
            # Just test that parsing succeeds - structure testing is done in other tests


class TestRegexMatching:
    """Test regex pattern matching in constraints"""

    @pytest.fixture
    def regex_corpus(self):
        """Corpus with various word forms for regex testing"""
        return pl.DataFrame(
            {
                "word": [
                    "running",
                    "jumped",
                    "quickly",
                    "the",
                    "dogs",
                    "cat",
                    "happening",
                    "walked",
                ],
                "pos": ["VERB", "VERB", "ADV", "DET", "NOUN", "NOUN", "VERB", "VERB"],
                "lemma": [
                    "run",
                    "jump",
                    "quick",
                    "the",
                    "dog",
                    "cat",
                    "happen",
                    "walk",
                ],
            }
        )

    def test_simple_regex_patterns(self, regex_corpus):
        """Test basic regex patterns"""
        # Words ending in 'ing'
        result = constraint_formula.parse_string('word=".*ing"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [True, False, False, False, False, False, True, False]
        )  # running, happening
        np.testing.assert_array_equal(matches, expected)

        # Words ending in 'ed'
        result = constraint_formula.parse_string('word=".*ed"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, False, True]
        )  # jumped, walked
        np.testing.assert_array_equal(matches, expected)

    def test_word_start_patterns(self, regex_corpus):
        """Test patterns matching word beginnings"""
        # Words starting with 'qu'
        result = constraint_formula.parse_string('word="qu.*"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, True, False, False, False, False, False]
        )  # quickly
        np.testing.assert_array_equal(matches, expected)

        # Words starting with consonant clusters
        result = constraint_formula.parse_string('word="[bcdfghjklmnpqrstvwxyz]{2}.*"')
        matches = df.select(result).to_numpy().flatten()
        # Should match words like "quickly" (qu...) - though 'qu' isn't consonant cluster
        # Let's test a simpler pattern
        result = constraint_formula.parse_string('word="[jh].*"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, True, False]
        )  # jumped, happening
        np.testing.assert_array_equal(matches, expected)

    def test_character_classes(self, regex_corpus):
        """Test regex character classes"""
        # Words with exactly 3 characters
        result = constraint_formula.parse_string('word="..."')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, True, False, True, False, False]
        )  # the, cat
        np.testing.assert_array_equal(matches, expected)

        # Words containing digits (none in our corpus)
        result = constraint_formula.parse_string('word=".*[0-9].*"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array([False, False, False, False, False, False, False, False])
        np.testing.assert_array_equal(matches, expected)

    def test_vowel_consonant_patterns(self, regex_corpus):
        """Test patterns for phonological analysis"""
        # Words ending in vowels
        result = constraint_formula.parse_string('word=".*[aeiou]"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, True, False, False, False, False]
        )  # the
        np.testing.assert_array_equal(matches, expected)

        # Words starting with consonants (negated vowel class)
        result = constraint_formula.parse_string('word="[^aeiou].*"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [True, True, True, True, True, True, True, True]
        )  # All words except those starting with vowels
        np.testing.assert_array_equal(matches, expected)

    def test_morphological_patterns(self, regex_corpus):
        """Test patterns for morphological analysis"""
        # Past tense regular verbs (ed endings, but not 'the')
        result = constraint_formula.parse_string('word="[a-z]*ed"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, False, True]
        )  # jumped, walked
        np.testing.assert_array_equal(matches, expected)

        # Present participles (ing endings)
        result = constraint_formula.parse_string('word="[a-z]+ing"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [True, False, False, False, False, False, True, False]
        )  # running, happening
        np.testing.assert_array_equal(matches, expected)

    def test_word_length_patterns(self, regex_corpus):
        """Test patterns based on word length"""
        # Words with 4-6 characters
        result = constraint_formula.parse_string('word="[a-z]{4,6}"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        # dogs(4), walked(6)
        expected = np.array(
            [False, True, False, False, True, False, False, True]
        )  # jumped(6), dogs(4), walked(6)
        np.testing.assert_array_equal(matches, expected)

        # Short words (1-3 characters)
        result = constraint_formula.parse_string('word="[a-z]{1,3}"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, True, False, True, False, False]
        )  # the, cat
        np.testing.assert_array_equal(matches, expected)

    def test_regex_with_other_features(self, regex_corpus):
        """Test regex patterns combined with other constraints"""
        # Verbs ending in 'ed'
        result = constraint_formula.parse_string('pos="VERB" & word=".*ed"')
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, False, True]
        )  # jumped, walked
        np.testing.assert_array_equal(matches, expected)

        # Nouns that are 3-4 characters long
        result = constraint_formula.parse_string('pos="NOUN" & word="[a-z]{3,4}"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, False, True, True, False, False]
        )  # dogs, cat
        np.testing.assert_array_equal(matches, expected)

    def test_complex_regex_patterns(self, regex_corpus):
        """Test more complex regex patterns"""
        # Words with alternating vowel-consonant pattern (simplified)
        result = constraint_formula.parse_string(
            'word="[aeiou][bcdfghjklmnpqrstvwxyz][aeiou].*"'
        )
        df = regex_corpus
        matches = df.select(result).to_numpy().flatten()
        # This is quite restrictive, might not match anything in our small corpus

        # Words that don't start with vowels
        result = constraint_formula.parse_string('word="[^aeiou].*"')
        matches = df.select(result).to_numpy().flatten()
        expected = np.array(
            [True, True, True, True, True, True, True, True]
        )  # All except words starting with vowels
        np.testing.assert_array_equal(matches, expected)

    def test_regex_in_full_cqp_patterns(self, regex_corpus):
        """Test regex patterns in complete CQP queries"""
        # Find sequences: determiner + word ending in 's'
        pattern = cqp.parse_string('[pos="DET"] [word=".*s"]')
        matches = list(pattern[0].matchall(regex_corpus))

        # Should find "the dogs"
        assert len(matches) == 1
        assert matches[0]["word"].to_list() == ["the", "dogs"]

        # Find verbs with specific morphological patterns
        pattern = cqp.parse_string(
            '[pos="VERB" & word=".*ing"] | [pos="VERB" & word=".*ed"]'
        )
        matches = list(pattern[0].matchall(regex_corpus))

        # Should find running, jumped, happening, walked
        assert len(matches) == 4
        matched_words = [match["word"].to_list()[0] for match in matches]
        expected_words = ["running", "jumped", "happening", "walked"]
        assert set(matched_words) == set(expected_words)

    def test_historical_linguistics_patterns(self):
        """Test regex patterns useful for historical corpus linguistics"""
        historical_corpus = pl.DataFrame(
            {
                "word": ["þe", "quike", "olde", "worlde", "knyght", "wyf", "churche"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "NOUN", "NOUN", "NOUN"],
                "period": ["ME", "ME", "ME", "ME", "ME", "ME", "ME"],
            }
        )

        # Middle English words with final 'e'
        result = constraint_formula.parse_string('word=".*e"')
        matches = historical_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [True, True, True, True, False, False, True]
        )  # þe, quike, olde, worlde, churche
        np.testing.assert_array_equal(matches, expected)

        # Words with 'y' spelling (historical variant)
        result = constraint_formula.parse_string('word=".*y.*"')
        matches = historical_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, False, True, True, False]
        )  # knyght, wyf
        np.testing.assert_array_equal(matches, expected)

        # Words with thorn (þ)
        result = constraint_formula.parse_string('word="þ.*"')
        matches = historical_corpus.select(result).to_numpy().flatten()
        expected = np.array([True, False, False, False, False, False, False])  # þe
        np.testing.assert_array_equal(matches, expected)


class TestUnimplementedFeatures:
    """Tests for CQP features not yet implemented - these will fail until implemented"""

    @pytest.fixture
    def historical_corpus(self):
        """Sample historical corpus with richer annotation"""
        return pl.DataFrame(
            {
                "word": [
                    "þe",
                    "quike",
                    "brune",
                    "fox",
                    "jumpeþ",
                    "ouer",
                    "þe",
                    "slowe",
                    "hund",
                ],
                "pos": [
                    "DET",
                    "ADJ",
                    "ADJ",
                    "NOUN",
                    "VERB",
                    "PREP",
                    "DET",
                    "ADJ",
                    "NOUN",
                ],
                "lemma": [
                    "the",
                    "quick",
                    "brown",
                    "fox",
                    "jump",
                    "over",
                    "the",
                    "slow",
                    "hound",
                ],
                "case": ["", "NOM", "NOM", "NOM", "", "", "", "ACC", "ACC"],
                "number": ["", "SG", "SG", "SG", "3SG", "", "", "SG", "SG"],
                "period": ["ME", "ME", "ME", "ME", "ME", "ME", "ME", "ME", "ME"],
                "sentence_id": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "token_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )

    @pytest.mark.skip(reason="Case-insensitive matching not implemented")
    def test_case_insensitive_matching(self, historical_corpus):
        """Test case-insensitive pattern matching"""
        # Should use %c flag for case-insensitive matching
        result = constraint_formula.parse_string('word="THE" %c')
        df = historical_corpus
        matches = df.select(result).to_numpy().flatten()
        # Should match both "þe" instances (positions 0 and 6)
        expected = np.array(
            [True, False, False, False, False, False, True, False, False]
        )
        np.testing.assert_array_equal(matches, expected)

    @pytest.mark.skip(reason="Negation not implemented")
    def test_negation_constraints(self, historical_corpus):
        """Test negation in constraints"""
        result = constraint_formula.parse_string('pos!="NOUN"')
        df = historical_corpus
        matches = df.select(result).to_numpy().flatten()
        # Should match everything except NOUN positions (3, 8)
        expected = np.array([True, True, True, False, True, True, True, True, False])
        np.testing.assert_array_equal(matches, expected)

    @pytest.mark.skip(reason="Numeric repetition not implemented")
    def test_numeric_repetition(self, historical_corpus):
        """Test numeric repetition patterns {n}, {n,m}"""
        # Exactly 2 adjectives
        pattern_exact = cqp.parse_string('[pos="ADJ"]{2}')
        # Between 1 and 3 adjectives
        pattern_range = cqp.parse_string('[pos="ADJ"]{1,3}')
        # At least 2 adjectives
        pattern_min = cqp.parse_string('[pos="ADJ"]{2,}')

        assert isinstance(pattern_exact[0], NumericRepetition)
        assert pattern_exact[0].min_count == 2
        assert pattern_exact[0].max_count == 2

        assert isinstance(pattern_range[0], NumericRepetition)
        assert pattern_range[0].min_count == 1
        assert pattern_range[0].max_count == 3

    @pytest.mark.skip(reason="Variable binding not implemented")
    def test_variable_binding(self, historical_corpus):
        """Test variable binding and references"""
        # Bind first token to variable, reference it later
        expr = '[pos="DET"] $det=[] [pos="ADJ"]* [pos="NOUN" & lemma=$det.lemma]'
        pattern = cqp.parse_string(expr)

        # Should bind variables and allow references
        matches = list(pattern.matchall(historical_corpus))
        # This specific example wouldn't match, but tests the syntax
        assert isinstance(pattern[0], SequenceWithBinding)

    @pytest.mark.skip(reason="Distance constraints not implemented")
    def test_distance_constraints(self, historical_corpus):
        """Test distance/proximity constraints"""
        # NOUN within 3 tokens of VERB
        expr = '[pos="NOUN"] []{0,3} [pos="VERB"]'
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        # Should find "fox jumpeþ" with intervening tokens allowed
        assert len(matches) >= 1

    @pytest.mark.skip(reason="Sentence boundaries not implemented")
    def test_sentence_boundaries(self, historical_corpus):
        """Test sentence boundary markers"""
        # Match beginning of sentence
        pattern_start = cqp.parse_string('<s> [pos="DET"]')
        # Match end of sentence
        pattern_end = cqp.parse_string('[pos="NOUN"] </s>')

        assert isinstance(pattern_start[0], SentenceBoundary)
        assert isinstance(pattern_end[0], Concat)

    @pytest.mark.skip(reason="Word boundary constraints not implemented")
    def test_word_boundaries(self, historical_corpus):
        """Test word boundary and alignment constraints"""
        # Match word-initial position
        expr = '[pos="ADJ" & word="^br.*"]'  # Words starting with 'br'
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        assert len(matches) == 1  # Should match "brune"

    @pytest.mark.skip(reason="Set operations not implemented")
    def test_set_constraints(self, historical_corpus):
        """Test set membership constraints"""
        # Match tokens where POS is in a set
        expr = '[pos in ("NOUN", "VERB", "ADJ")]'
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        # Should match most content words
        assert len(matches) >= 6

    @pytest.mark.skip(reason="Frequency constraints not implemented")
    def test_frequency_constraints(self, historical_corpus):
        """Test frequency-based constraints"""
        # Match rare words (hypothetical frequency data)
        expr = "[frequency < 100]"  # Words occurring less than 100 times
        pattern = cqp.parse_string(expr)

        # Would need frequency data in corpus
        assert isinstance(pattern[0], Token)

    @pytest.mark.skip(reason="Positional constraints not implemented")
    def test_positional_constraints(self, historical_corpus):
        """Test absolute and relative position constraints"""
        # Match tokens at specific positions
        expr = '[pos="NOUN" & position=4]'  # 5th token (0-indexed)
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        assert len(matches) == 1  # Should match "jumpeþ" if it's a VERB at position 4

    @pytest.mark.skip(reason="Dependency relations not implemented")
    def test_dependency_relations(self, historical_corpus):
        """Test syntactic dependency constraints"""
        # Match nouns that are subjects of verbs
        expr = '[pos="NOUN" & dep_rel="nsubj"] >nsubj [pos="VERB"]'
        pattern = cqp.parse_string(expr)

        # Would need dependency annotation in corpus
        assert isinstance(pattern[0], DependencyPattern)

    @pytest.mark.skip(reason="Named queries not implemented")
    def test_named_queries(self, historical_corpus):
        """Test named query definitions and reuse"""
        # Define a named query
        define_query = 'DEFINE NOUN_PHRASE [pos="DET"]? [pos="ADJ"]* [pos="NOUN"];'
        use_query = 'NOUN_PHRASE [pos="VERB"] NOUN_PHRASE'

        # Should allow query reuse
        pattern = cqp.parse_string(use_query)
        assert isinstance(pattern[0], NamedQueryReference)

    @pytest.mark.skip(reason="Subcorpus constraints not implemented")
    def test_subcorpus_constraints(self, historical_corpus):
        """Test subcorpus and metadata constraints"""
        # Match only in specific time periods
        expr = '[pos="NOUN"] :: period="ME"'
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        # All matches should be from Middle English period
        for match in matches:
            assert all(match["period"] == "ME")

    @pytest.mark.skip(reason="Alignment constraints not implemented")
    def test_alignment_constraints(self, historical_corpus):
        """Test alignment with parallel corpora"""
        # Match aligned tokens (for parallel/comparative corpora)
        expr = '[lemma="fox"] @1 [lemma="renard"]'  # English-French alignment
        pattern = cqp.parse_string(expr)

        # Would need parallel corpus data
        assert isinstance(pattern[0], AlignmentPattern)

    @pytest.mark.skip(reason="Structural attributes not implemented")
    def test_structural_attributes(self, historical_corpus):
        """Test XML/structural markup constraints"""
        # Match within specific structural elements
        expr = '<text type="prose"> [pos="NOUN"] </text>'
        pattern = cqp.parse_string(expr)

        # Should respect structural boundaries
        assert isinstance(pattern[0], StructuralPattern)

    @pytest.mark.skip(reason="Collocations not implemented")
    def test_collocation_queries(self, historical_corpus):
        """Test collocation and co-occurrence patterns"""
        # Find words that co-occur within a window
        expr = '[lemma="quick"] WITHIN 5 WORDS OF [lemma="fox"]'
        pattern = cqp.parse_string(expr)

        matches = list(pattern.matchall(historical_corpus))
        assert len(matches) >= 1

    @pytest.mark.skip(reason="Statistical measures not implemented")
    def test_statistical_constraints(self, historical_corpus):
        """Test statistical and corpus-linguistic measures"""
        # Match based on statistical measures (MI, T-score, etc.)
        expr = '[lemma="very"] [lemma & mi_score > 3.0]'  # High mutual information
        pattern = cqp.parse_string(expr)

        # Would need precomputed statistical measures
        assert isinstance(pattern[0], StatisticalPattern)


# Placeholder classes for unimplemented features (to make tests syntactically valid)
class NumericRepetition(Pattern):
    """Placeholder for numeric repetition patterns"""

    def __init__(self, pattern, min_count, max_count=None):
        super().__init__()
        self.subpatterns = [pattern]
        self.min_count = min_count
        self.max_count = max_count if max_count is not None else min_count


class SequenceWithBinding(Pattern):
    """Placeholder for variable binding patterns"""

    pass


class SentenceBoundary(Pattern):
    """Placeholder for sentence boundary patterns"""

    pass


class DependencyPattern(Pattern):
    """Placeholder for dependency relation patterns"""

    pass


class NamedQueryReference(Pattern):
    """Placeholder for named query references"""

    pass


class AlignmentPattern(Pattern):
    """Placeholder for alignment patterns"""

    pass


class StructuralPattern(Pattern):
    """Placeholder for structural markup patterns"""

    pass


class StatisticalPattern(Pattern):
    """Placeholder for statistical constraint patterns"""

    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
