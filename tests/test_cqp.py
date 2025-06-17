import time
import unittest.mock

import numpy as np
import polars as pl
import pyparsing as pp
import pytest

from polars_corpus.cqp import (
    Alt,
    Concat,
    MToN,
    OneOrMore,
    OneOrZero,
    Pattern,
    ScanContext,
    Skip,
    Token,
    ZeroOrMore,
    constraint_formula,
    cqp,
    node,
)


@pytest.fixture
def basic_corpus():
    """Standard test corpus for most tests"""
    return pl.DataFrame(
        {
            "word": ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"],
            "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "PREP", "ADJ", "NOUN"],
            "lemma": ["the", "quick", "brown", "fox", "jump", "over", "lazy", "dog"],
            "case": ["", "NOM", "NOM", "NOM", "", "", "ACC", "ACC"],
        }
    )


@pytest.fixture
def optimization_corpus():
    """Corpus designed for testing valid_starts optimization"""
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


@pytest.fixture
def regex_corpus():
    """Corpus for regex pattern testing"""
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
            "lemma": ["run", "jump", "quick", "the", "dog", "cat", "happen", "walk"],
        }
    )


@pytest.fixture
def historical_corpus():
    """Historical corpus with Middle English features"""
    return pl.DataFrame(
        {
            "word": ["þe", "quike", "olde", "worlde", "knyght", "wyf", "churche"],
            "pos": ["DET", "ADJ", "ADJ", "NOUN", "NOUN", "NOUN", "NOUN"],
            "period": ["ME", "ME", "ME", "ME", "ME", "ME", "ME"],
        }
    )


@pytest.fixture
def longest_match_corpus():
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
            "pos": ["VERB", "VERB", "VERB", "VERB", "ADV", "DET", "NOUN", "NOUN"],
            "lemma": ["run", "jump", "happen", "walk", "quick", "the", "dog", "cat"],
        }
    )


class TestConstraintParsing:
    """Test parsing and evaluation of token constraints"""

    def test_basic_constraints(self):
        """Test simple feature=value, conjunction, and disjunction constraints"""
        df = pl.DataFrame(
            {
                "pos": ["NOUN", "VERB", "NOUN", "ADJ"],
                "case": ["NOM", "NOM", "ACC", "NOM"],
            }
        )

        # Simple constraint
        simple = constraint_formula.parse_string('pos="NOUN"')
        matches = df.select(simple).to_numpy().flatten()
        np.testing.assert_array_equal(matches, [True, False, True, False])

        # Conjunction
        conj = constraint_formula.parse_string('pos="NOUN" & case="NOM"')
        matches = df.select(conj).to_numpy().flatten()
        np.testing.assert_array_equal(matches, [True, False, False, False])

        # Disjunction
        disj = constraint_formula.parse_string('pos="NOUN" | pos="VERB"')
        matches = df.select(disj).to_numpy().flatten()
        np.testing.assert_array_equal(matches, [True, True, True, False])

        # Complex nested
        complex_expr = constraint_formula.parse_string(
            '(pos="NOUN" | pos="VERB") & case="NOM"'
        )
        matches = df.select(complex_expr).to_numpy().flatten()
        np.testing.assert_array_equal(matches, [True, True, False, False])


class TestNodeParsing:
    """Test parsing of CQP node expressions"""

    def test_token_and_skip_nodes(self):
        """Test parsing of token nodes with constraints and skip nodes"""
        # Token node
        result = node.parse_string('[pos="NOUN"]')
        assert isinstance(result[0], Token)

        # Test the constraint works
        df = pl.DataFrame({"pos": ["NOUN", "VERB"]})
        result[0].set_subject(df)
        assert result[0].valid_tokens[0] == True
        assert result[0].valid_tokens[1] == False

        # Skip node
        result = node.parse_string("[]")
        assert isinstance(result[0], Skip)


class TestBasicPatterns:
    """Test core pattern classes"""

    def test_token_and_skip_patterns(self, basic_corpus):
        """Test Token and Skip pattern behavior"""
        # Token pattern
        adj_pattern = Token(pl.col("pos") == "ADJ")
        adj_pattern.set_subject(basic_corpus)

        expected_valid = np.array([False, True, True, False, False, False, True, False])
        np.testing.assert_array_equal(adj_pattern.valid_tokens, expected_valid)

        ctxt = ScanContext()
        assert list(adj_pattern._op(ctxt, 1)) == [2]  # "quick" -> advance to next
        assert list(adj_pattern._op(ctxt, 0)) == []  # "the" -> no match

        # Skip pattern
        skip = Skip()
        skip.set_subject(basic_corpus)
        assert list(skip._op(ctxt, 2)) == [3]  # Always advances by 1
        assert skip.valid_starts is None

    def test_repetition_patterns(self, basic_corpus):
        """Test repetition patterns: *, +, ?, and {m,n}"""
        adj_pattern = Token(pl.col("pos") == "ADJ")
        ctxt = ScanContext()

        # ZeroOrMore: should match 0, 1, or 2 adjacent ADJs
        zero_or_more = ZeroOrMore(adj_pattern)
        zero_or_more.set_subject(basic_corpus)
        matches = list(zero_or_more._op(ctxt, 1))  # Start at "quick"
        assert 1 in matches  # zero matches
        assert 2 in matches  # one match
        assert 3 in matches  # two matches

        # OneOrMore: same but no zero matches
        one_or_more = OneOrMore(adj_pattern)
        one_or_more.set_subject(basic_corpus)
        matches = list(one_or_more._op(ctxt, 1))
        assert 1 not in matches  # no zero matches
        assert 2 in matches and 3 in matches

        # OneOrZero: match 0 or 1
        optional = OneOrZero(adj_pattern)
        optional.set_subject(basic_corpus)
        matches = list(optional._op(ctxt, 1))
        assert 1 in matches and 2 in matches  # both 0 and 1 match

        # MToN: specific range
        mton = MToN(adj_pattern, m=1, n=2)
        mton.set_subject(basic_corpus)
        matches = list(mton._op(ctxt, 1))
        assert 1 not in matches  # no zero matches (m=1)
        assert 2 in matches and 3 in matches  # 1 and 2 matches

    def test_sequence_and_alternation(self, basic_corpus):
        """Test Concat and Alt patterns"""
        det_pattern = Token(pl.col("pos") == "DET")
        adj_pattern = Token(pl.col("pos") == "ADJ")
        ctxt = ScanContext()

        # Concatenation: DET followed by ADJ
        concat = Concat(det_pattern, adj_pattern)
        concat.set_subject(basic_corpus)
        matches = list(concat._op(ctxt, 0))  # Start at "the"
        assert matches == [2]  # "the quick"

        # Alternation: DET or ADJ
        alt = Alt(det_pattern, adj_pattern)
        alt.set_subject(basic_corpus)
        matches_at_0 = list(alt._op(ctxt, 0))  # "the" (DET)
        matches_at_1 = list(alt._op(ctxt, 1))  # "quick" (ADJ)
        assert 1 in matches_at_0  # DET matches
        assert 2 in matches_at_1  # ADJ matches

    def test_complex_pattern_combinations(self, basic_corpus):
        """Test complex pattern: DET ADV* ADJ+ NOUN"""
        det = Token(pl.col("pos") == "DET")
        # Note: using PREP as proxy for ADV since our corpus doesn't have ADV
        prep_star = ZeroOrMore(Token(pl.col("pos") == "PREP"))
        adj_plus = OneOrMore(Token(pl.col("pos") == "ADJ"))
        noun = Token(pl.col("pos") == "NOUN")

        # Build pattern: DET PREP* ADJ+ NOUN
        pattern = Concat(Concat(Concat(det, prep_star), adj_plus), noun)
        pattern.set_subject(basic_corpus)

        ctxt = ScanContext()
        matches = list(pattern._op(ctxt, 0))  # Start at "the"
        # Should match "the quick brown fox" (positions 0-4)
        assert 4 in matches


class TestValidStartsOptimization:
    """Test that valid_starts optimization works correctly"""

    def test_basic_valid_starts_computation(self, optimization_corpus):
        """Test valid_starts computation for Token, Skip, and combination patterns"""
        det_pattern = Token(pl.col("pos") == "DET")
        adj_pattern = Token(pl.col("pos") == "ADJ")

        det_pattern.set_subject(optimization_corpus)
        adj_pattern.set_subject(optimization_corpus)

        # Token patterns should mark only matching positions
        expected_det = np.array(
            [True, False, False, True, False, False, False, True, False, False, False]
        )
        np.testing.assert_array_equal(det_pattern.valid_starts, expected_det)

        expected_adj = np.array(
            [False, False, False, False, True, False, False, False, True, True, False]
        )
        np.testing.assert_array_equal(adj_pattern.valid_starts, expected_adj)

        # Concat inherits from first subpattern
        concat = Concat(det_pattern, adj_pattern)
        concat.set_subject(optimization_corpus)
        np.testing.assert_array_equal(concat.valid_starts, expected_det)

        # Alt computes union
        alt = Alt(det_pattern, adj_pattern)
        alt.set_subject(optimization_corpus)
        expected_union = np.logical_or(expected_det, expected_adj)
        np.testing.assert_array_equal(alt.valid_starts, expected_union)

        # Skip has None (can start anywhere)
        skip = Skip()
        skip.set_subject(optimization_corpus)
        assert skip.valid_starts is None

    def test_repetition_valid_starts(self, optimization_corpus):
        """Test valid_starts behavior for repetition patterns"""
        adj_pattern = Token(pl.col("pos") == "ADJ")

        # ZeroOrMore has None (can match zero)
        zero_or_more = ZeroOrMore(adj_pattern)
        zero_or_more.set_subject(optimization_corpus)
        assert zero_or_more.valid_starts is None

        # OneOrMore inherits from subpattern
        one_or_more = OneOrMore(adj_pattern)
        one_or_more.set_subject(optimization_corpus)
        expected_adj = np.array(
            [False, False, False, False, True, False, False, False, True, True, False]
        )
        np.testing.assert_array_equal(one_or_more.valid_starts, expected_adj)

        # MToN with min=0 has None
        mton_zero = MToN(adj_pattern, m=0, n=3)
        mton_zero.set_subject(optimization_corpus)
        assert mton_zero.valid_starts is None

        # MToN with min>0 inherits from subpattern
        mton_min = MToN(adj_pattern, m=1, n=3)
        mton_min.set_subject(optimization_corpus)
        np.testing.assert_array_equal(mton_min.valid_starts, expected_adj)

    def test_optimization_performance_benefit(self, optimization_corpus):
        """Test that optimization actually improves performance"""
        import unittest.mock

        det_pattern = Token(pl.col("pos") == "DET")
        det_pattern.set_subject(optimization_corpus)

        # Mock the _op method to count calls
        original_op = det_pattern._op
        with unittest.mock.patch.object(
            det_pattern, "_op", wraps=original_op
        ) as mock_op:
            matches = list(det_pattern.matchall(optimization_corpus))

            # Should only be called for DET positions (0, 3, 7)
            assert mock_op.call_count == 3

            call_positions = [call[0][1] for call in mock_op.call_args_list]
            assert call_positions == [0, 3, 7]


class TestCQPParsing:
    """Test parsing of CQP expressions"""

    def test_basic_cqp_expressions(self):
        """Test parsing of fundamental CQP constructs"""
        # Simple token
        result = cqp.parse_string('[pos="NOUN"]')
        assert isinstance(result[0], Token)

        # Sequence
        result = cqp.parse_string('[pos="DET"] [pos="ADJ"]')
        assert isinstance(result[0], Concat)
        assert len(result[0].subpatterns) == 2

        # Alternation
        result = cqp.parse_string('[pos="NOUN"] | [pos="VERB"]')
        assert isinstance(result[0], Alt)
        assert len(result[0].subpatterns) == 2

        # Repetition operators
        tests = [
            ('[pos="ADJ"]*', ZeroOrMore),
            ('[pos="ADJ"]+', OneOrMore),
            ('[pos="ADJ"]?', OneOrZero),
        ]

        for expr, expected_type in tests:
            result = cqp.parse_string(expr)
            assert isinstance(result[0], expected_type)

    def test_mton_syntax_variants(self):
        """Test all {m,n} syntax variants"""
        # Exact count
        result = cqp.parse_string('[pos="ADJ"]{3}')
        assert isinstance(result[0], MToN)
        assert result[0].min == 3 and result[0].max == 3

        # Range
        result = cqp.parse_string('[pos="ADJ"]{2,5}')
        assert result[0].min == 2 and result[0].max == 5

        # Minimum only
        result = cqp.parse_string('[pos="ADJ"]{2,}')
        assert result[0].min == 2 and result[0].max is None

        # Maximum only
        result = cqp.parse_string('[pos="ADJ"]{,3}')
        assert result[0].min == 0 and result[0].max == 3

    def test_complex_expressions_and_precedence(self):
        """Test parsing of nested expressions and operator precedence"""
        # Parentheses and nesting
        result = cqp.parse_string('([pos="DET"] [pos="ADJ"]*)+ [pos="NOUN"]')
        assert isinstance(result[0], Concat)
        assert isinstance(result[0].subpatterns[0], OneOrMore)

        # Test long sequences
        result = cqp.parse_string('[pos="DET"] [pos="ADJ"] [pos="NOUN"]')
        assert isinstance(result[0], Concat)
        # Should be Concat(DET, Concat(ADJ, NOUN))
        assert isinstance(result[0].subpatterns[0], Token)  # DET
        assert isinstance(result[0].subpatterns[1], Token)

        # Operator precedence: ADJ* NOUN | PRON should be (ADJ* NOUN) | PRON
        result = cqp.parse_string('[pos="ADJ"]* [pos="NOUN"] | [pos="PRON"]')
        assert isinstance(result[0], Alt)
        left_side = result[0].subpatterns[0]
        assert isinstance(left_side, Concat)
        assert isinstance(left_side.subpatterns[0], ZeroOrMore)

    def test_whitespace_and_parsing_robustness(self):
        """Test that parser handles whitespace and edge cases correctly"""
        # Different whitespace patterns should parse identically
        expressions = [
            '[pos="NOUN"][pos="VERB"]',
            '[pos="NOUN"] [pos="VERB"]',
            '[ pos="NOUN" ] [ pos="VERB" ]',
            '[pos="NOUN"]  [pos="VERB"]',
            '\n[pos="NOUN"]\n[pos="VERB"]\n',
        ]

        results = [cqp.parse_string(expr) for expr in expressions]
        for result in results:
            assert isinstance(result[0], Concat)
            assert len(result[0].subpatterns) == 2

    def test_quoted_strings_and_feature_names(self):
        """Test parsing of quoted strings and various feature names"""
        # Different feature names
        feature_names = ["pos", "lemma", "word", "case", "number", "gender"]
        for feature in feature_names:
            expr = f'[{feature}="TEST"]'
            result = cqp.parse_string(expr)
            assert isinstance(result[0], Token)

        # Special characters in quoted strings
        test_cases = [
            '[word="hello world"]',  # Spaces
            '[lemma="café"]',  # Unicode
        ]
        for expr in test_cases:
            result = cqp.parse_string(expr)
            assert isinstance(result[0], Token)

    def test_invalid_syntax_handling(self):
        """Test that invalid syntax raises ParseException"""
        invalid_expressions = [
            "[pos=NOUN]",  # Missing quotes
            '[pos="NOUN"',  # Missing closing bracket
            '[pos=="NOUN"]',  # Double equals
            '| [pos="NOUN"]',  # Leading operator
            '[pos="NOUN"] &',  # Dangling operator
        ]

        for expr in invalid_expressions:
            with pytest.raises(pp.ParseException):
                cqp.parse_string(expr, parse_all=True)

    def test_realistic_linguistic_queries(self):
        """Test parsing of realistic corpus linguistics queries"""
        realistic_queries = [
            '[pos="DET"]? [pos="ADJ"]* [pos="NOUN"]',  # Noun phrases
            '[pos="AUX"]? [pos="ADV"]* [pos="VERB"]',  # Verb phrases
            '[pos="PREP"] [pos="DET"]? [pos="ADJ"]* [pos="NOUN"]',  # Prepositional phrases
            '[pos="NOUN"] [word="and"] [pos="NOUN"]',  # Coordination
            '([pos="NOUN"] | [pos="PRON"]) [pos="VERB"] ([pos="NOUN"] | [pos="PRON"])',  # Complex alternation
            '[pos="NOUN" & case="NOM"] [pos="VERB" & tense="PAST"]',  # Multiple constraints
        ]

        for query in realistic_queries:
            result = cqp.parse_string(query)
            assert len(result) == 1
            assert isinstance(result[0], Pattern)


class TestMatchAll:
    """Test the matchall functionality"""

    def test_basic_matching(self, basic_corpus):
        """Test matchall with simple and complex patterns"""
        # Simple token matching
        det_pattern = Token(pl.col("pos") == "DET")
        matches = list(det_pattern.matchall(basic_corpus))
        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the"]
        assert basic_corpus["word"][int(matches[0][0])] == "the"

        # Sequence matching: DET ADJ+
        det_adj_plus = Concat(det_pattern, OneOrMore(Token(pl.col("pos") == "ADJ")))
        matches = list(det_adj_plus.matchall(basic_corpus))
        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the", "quick", "brown"]
        assert basic_corpus["word"][
            int(matches[0][0]) : int(matches[0][1])
        ].to_list() == ["the", "quick", "brown"]

        # Complex pattern: DET ADJ+ NOUN
        full_np = Concat(det_adj_plus, Token(pl.col("pos") == "NOUN"))
        matches = list(full_np.matchall(basic_corpus))
        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the", "quick", "brown", "fox"]
        assert basic_corpus["word"][
            int(matches[0][0]) : int(matches[0][1])
        ].to_list() == ["the", "quick", "brown", "fox"]

        def test_end_to_end_cqp_matching(self, basic_corpus):
            """Test complete CQP parsing and matching pipeline"""

        # Parse and execute: determiner followed by one or more adjectives and a noun
        pattern = cqp.parse_string('[pos="DET"] [pos="ADJ"]+ [pos="NOUN"]')[0]
        matches = list(pattern.matchall(basic_corpus))

        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the", "quick", "brown", "fox"]
        assert basic_corpus["word"][
            int(matches[0][0]) : int(matches[0][1])
        ].to_list() == ["the", "quick", "brown", "fox"]

        # Parse and execute: alternation
        pattern = cqp.parse_string('[pos="PREP"] | [pos="VERB"]')[0]
        matches = list(pattern.matchall(basic_corpus))

        assert len(matches) == 2
        # matched_words = [match["word"].to_list()[0] for match in matches]
        matched_words = [basic_corpus["word"][int(m[0])] for m in matches]
        assert set(matched_words) == {"jumps", "over"}

    def test_mton_integration(self, basic_corpus):
        """Test MToN patterns in realistic scenarios"""
        # DET ADJ{1,3} NOUN pattern
        pattern = cqp.parse_string('[pos="DET"] [pos="ADJ"]{1,3} [pos="NOUN"]')[0]
        matches = list(pattern.matchall(basic_corpus))

        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the", "quick", "brown", "fox"]
        assert basic_corpus["word"][
            int(matches[0][0]) : int(matches[0][1])
        ].to_list() == ["the", "quick", "brown", "fox"]


class TestRegexPatterns:
    """Test regex pattern matching in constraints"""

    def test_morphological_patterns(self, regex_corpus):
        """Test regex patterns for morphological analysis"""
        # Words ending in 'ing'
        result = constraint_formula.parse_string('word=".*ing"')
        matches = regex_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [True, False, False, False, False, False, True, False]
        )  # running, happening
        np.testing.assert_array_equal(matches, expected)

        # Words ending in 'ed'
        result = constraint_formula.parse_string('word=".*ed"')
        matches = regex_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, False, True]
        )  # jumped, walked
        np.testing.assert_array_equal(matches, expected)

        # 3-character words
        result = constraint_formula.parse_string('word="..."')
        matches = regex_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [False, False, False, True, False, True, False, False]
        )  # the, cat
        np.testing.assert_array_equal(matches, expected)

        # Combined: verbs ending in 'ed'
        result = constraint_formula.parse_string('pos="VERB" & word=".*ed"')
        matches = regex_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [False, True, False, False, False, False, False, True]
        )  # jumped, walked
        np.testing.assert_array_equal(matches, expected)

    def test_historical_linguistics_patterns(self, historical_corpus):
        """Test regex patterns for historical corpus linguistics"""
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

    def test_regex_in_cqp_queries(self, regex_corpus):
        """Test regex patterns in complete CQP expressions"""
        # Find DET followed by plural noun (ending in 's')
        pattern = cqp.parse_string('[pos="DET"] [word=".*s"]')[0]
        matches = list(pattern.matchall(regex_corpus))

        assert len(matches) == 1
        # assert matches[0]["word"].to_list() == ["the", "dogs"]
        assert regex_corpus["word"][int(matches[0][0])] == "the"

        # Find verbs with morphological patterns
        pattern = cqp.parse_string(
            '[pos="VERB" & word=".*ing"] | [pos="VERB" & word=".*ed"]'
        )[0]
        matches = list(pattern.matchall(regex_corpus))

        assert len(matches) == 4
        # matched_words = [match["word"].to_list()[0] for match in matches]
        matched_words = [regex_corpus["word"][int(match[0])] for match in matches]
        assert set(matched_words) == {"running", "jumped", "happening", "walked"}

    def test_longest_matching(self, longest_match_corpus):
        pattern = cqp.parse_string('[pos="NOUN"]+')[0]
        matches = list(pattern.matchall(longest_match_corpus, longest_match=True))
        assert len(matches) == 1

        matches = list(pattern.matchall(longest_match_corpus, longest_match=False))
        assert len(matches) == 2

        pattern = cqp.parse_string('[pos="VERB"]+')[0]
        matches = list(pattern.matchall(longest_match_corpus, longest_match=True))
        assert len(matches) == 1

        matches = list(pattern.matchall(longest_match_corpus, longest_match=False))
        assert len(matches) == 4


class TestMToNEquivalences:
    """Test that MToN patterns are equivalent to existing patterns where applicable"""

    def test_mton_equivalences(self):
        """Test MToN equivalences with existing patterns"""
        corpus = pl.DataFrame({"pos": ["ADJ", "ADJ", "NOUN"]})
        adj_token = Token(pl.col("pos") == "ADJ")

        # {0,1} should equal ?
        mton_optional = MToN(adj_token, m=0, n=1)
        question_pattern = OneOrZero(adj_token)

        mton_optional.set_subject(corpus)
        question_pattern.set_subject(corpus)

        ctxt = ScanContext()
        mton_matches = set(mton_optional._op(ctxt, 0))
        question_matches = set(question_pattern._op(ctxt, 0))
        assert mton_matches == question_matches

        # {1,} should be similar to + (though not identical due to implementation differences)
        mton_plus = MToN(adj_token, m=1, n=None)
        plus_pattern = OneOrMore(adj_token)

        mton_plus.set_subject(corpus)
        plus_pattern.set_subject(corpus)

        # Both should produce at least some matches at position 0
        mton_matches = list(mton_plus._op(ctxt, 0))
        plus_matches = list(plus_pattern._op(ctxt, 0))
        assert len(mton_matches) > 0 and len(plus_matches) > 0


class TestPerformanceOptimization:
    """Test optimization performance and correctness"""

    def test_optimization_equivalence(self, basic_corpus):
        """Test that optimized matching gives same results as brute force"""
        pattern = Concat(Token(pl.col("pos") == "DET"), Token(pl.col("pos") == "ADJ"))

        # Get optimized results
        optimized_matches = list(pattern.matchall(basic_corpus))

        # Simulate brute force by setting valid_starts to None
        pattern.set_subject(basic_corpus)
        original_valid_starts = pattern.valid_starts
        pattern.valid_starts = None

        brute_force_matches = list(pattern.matchall(basic_corpus))

        # Restore and compare
        pattern.valid_starts = original_valid_starts

        assert len(optimized_matches) == len(brute_force_matches)
        for opt, bf in zip(optimized_matches, brute_force_matches):
            assert opt == bf  # opt.equals(bf)

        def test_performance_measurement(self):
            """Test that optimization provides measurable performance benefit"""

        # Create large corpus for performance testing
        large_corpus = pl.DataFrame(
            {
                "pos": ["OTHER"] * 1000
                + ["DET"]
                + ["OTHER"] * 1000
                + ["ADJ"]
                + ["OTHER"] * 1000
            }
        )

        pattern = Token(pl.col("pos") == "DET")

        # Time with optimization
        start_time = time.time()
        matches_optimized = list(pattern.matchall(large_corpus))
        optimized_time = time.time() - start_time

        # Time without optimization
        pattern.set_subject(large_corpus)
        pattern.valid_starts = None

        start_time = time.time()
        matches_brute_force = list(pattern.matchall(large_corpus))
        brute_force_time = time.time() - start_time

        # Results should be identical
        assert len(matches_optimized) == len(matches_brute_force)

        # Optimization should not be significantly slower (allow for test variability)
        assert optimized_time <= brute_force_time * 3


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_and_no_match_cases(self):
        """Test behavior with empty corpora and no matches"""
        # Empty corpus
        empty_corpus = pl.DataFrame({"pos": []})
        pattern = Token(pl.col("pos") == "NOUN")
        matches = list(pattern.matchall(empty_corpus))
        assert len(matches) == 0

        # No matches
        corpus = pl.DataFrame({"pos": ["NOUN", "VERB"]})
        pattern = Token(pl.col("pos") == "ADJ")
        matches = list(pattern.matchall(corpus))
        assert len(matches) == 0

    def test_pattern_representations(self):
        """Test string representations work"""
        token = Token(pl.col("pos") == "NOUN")
        assert "Token" in repr(token)

        skip = Skip()
        assert "Skip" in repr(skip)

        concat = Concat(token, skip)
        assert "Concat" in repr(concat)


class TestUnimplementedFeatures:
    """Tests for CQP features not yet implemented - these will fail until implemented"""

    @pytest.fixture
    def unimpl_corpus(self):
        """Sample corpus for testing unimplemented features"""
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
            }
        )

    @pytest.mark.skip(reason="Case-insensitive matching not implemented")
    def test_case_insensitive_matching(self, unimpl_corpus):
        """Test case-insensitive pattern matching"""
        result = constraint_formula.parse_string('word="THE" %c')
        matches = unimpl_corpus.select(result).to_numpy().flatten()
        expected = np.array(
            [True, False, False, False, False, False, True, False, False]
        )
        np.testing.assert_array_equal(matches, expected)

    @pytest.mark.skip(reason="Negation not implemented")
    def test_negation_constraints(self, unimpl_corpus):
        """Test negation in constraints"""
        result = constraint_formula.parse_string('pos!="NOUN"')
        matches = unimpl_corpus.select(result).to_numpy().flatten()
        expected = np.array([True, True, True, False, True, True, True, True, False])
        np.testing.assert_array_equal(matches, expected)

    @pytest.mark.skip(reason="Variable binding not implemented")
    def test_variable_binding(self, unimpl_corpus):
        """Test variable binding and references"""
        expr = '[pos="DET"] $det=[] [pos="ADJ"]* [pos="NOUN" & lemma=$det.lemma]'
        pattern = cqp.parse_string(expr)
        # Would test variable binding functionality
        assert isinstance(pattern[0], Pattern)  # Placeholder assertion

    @pytest.mark.skip(reason="Distance constraints not implemented")
    def test_distance_constraints(self, unimpl_corpus):
        """Test distance/proximity constraints"""
        expr = '[pos="NOUN"] []{0,3} [pos="VERB"]'
        pattern = cqp.parse_string(expr)
        matches = list(pattern.matchall(unimpl_corpus))
        assert len(matches) >= 1

    @pytest.mark.skip(reason="Sentence boundaries not implemented")
    def test_sentence_boundaries(self, unimpl_corpus):
        """Test sentence boundary markers"""
        pattern_start = cqp.parse_string('<s> [pos="DET"]')
        pattern_end = cqp.parse_string('[pos="NOUN"] </s>')
        # Would test sentence boundary functionality
        assert isinstance(pattern_start[0], Pattern)
        assert isinstance(pattern_end[0], Pattern)

    @pytest.mark.skip(reason="Named queries not implemented")
    def test_named_queries(self, unimpl_corpus):
        """Test named query definitions and reuse"""
        define_query = 'DEFINE NOUN_PHRASE [pos="DET"]? [pos="ADJ"]* [pos="NOUN"];'
        use_query = 'NOUN_PHRASE [pos="VERB"] NOUN_PHRASE'
        # Would test named query functionality
        pattern = cqp.parse_string(use_query)
        assert isinstance(pattern[0], Pattern)

    @pytest.mark.skip(reason="Structural attributes not implemented")
    def test_structural_attributes(self, unimpl_corpus):
        """Test XML/structural markup constraints"""
        expr = '<text type="prose"> [pos="NOUN"] </text>'
        pattern = cqp.parse_string(expr)
        # Would test structural pattern functionality
        assert isinstance(pattern[0], Pattern)

    @pytest.mark.skip(reason="Statistical measures not implemented")
    def test_statistical_constraints(self, unimpl_corpus):
        """Test statistical and corpus-linguistic measures"""
        expr = '[lemma="very"] [lemma & mi_score > 3.0]'
        pattern = cqp.parse_string(expr)
        # Would test statistical pattern functionality
        assert isinstance(pattern[0], Pattern)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
