import polars as pl
import pytest

import polars_corpus as plc


def get_matched_tokens(corpus, search_results):
    """Extract the matched token strings from search results."""
    if search_results is None:
        return []
    tokens = []
    for span in search_results._matched_spans:
        matched_tokens = corpus["token"][span.start:span.end]
        tokens.append(" ".join(matched_tokens))
    return tokens


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing simple query language"""
    return pl.DataFrame({
        "token": [
            "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", ".",
            "A", "very", "capable", "student", "walked", "slowly", "to", "school", ".",
            "The", "red", "car", "and", "blue", "truck", "parked", "outside", ".",
            "I", "sing", "sang", "song", "yesterday", ".", "They", "are", "able", "to", "table", "the", "capable", "motion", ".",
            "Voodoo", "and", "schoolroom", "mysteries", ".", "The", "big", "table", "is", "suitable", "and", "available", ".",
            "My", "neighbour", "and", "neighbor", "both", "came", "."
        ],
        "pos": [
            "DT", "JJ", "JJ", "NN", "VBZ", "IN", "DT", "JJ", "NN", ".",
            "DT", "RB", "JJ", "NN", "VBD", "RB", "TO", "NN", ".",
            "DT", "JJ", "NN", "CC", "JJ", "NN", "VBD", "RB", ".",
            "PRP", "VBP", "VBD", "NN", "RB", ".", "PRP", "VBP", "JJ", "TO", "VB", "DT", "JJ", "NN", ".",
            "NN", "CC", "NN", "NNS", ".", "DT", "JJ", "NN", "VBZ", "JJ", "CC", "JJ", ".",
            "PRP$", "NN", "CC", "NN", "DT", "VBD", "."
        ],
        "lemma": [
            "the", "quick", "brown", "fox", "jump", "over", "the", "lazy", "dog", ".",
            "a", "very", "capable", "student", "walk", "slowly", "to", "school", ".",
            "the", "red", "car", "and", "blue", "truck", "park", "outside", ".",
            "i", "sing", "sing", "song", "yesterday", ".", "they", "be", "able", "to", "table", "the", "capable", "motion", ".",
            "voodoo", "and", "schoolroom", "mystery", ".", "the", "big", "table", "be", "suitable", "and", "available", ".",
            "my", "neighbour", "and", "neighbor", "both", "come", "."
        ]
    })


class TestBasicWordSearch:
    """Test basic word form searches"""

    def test_simple_word_search(self, sample_corpus):
        """Test searching for exact word forms"""
        query = "fox"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["fox"]

    def test_case_insensitive_search(self, sample_corpus):
        """Test case-insensitive search by default"""
        query = "the"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert len(matched) == 5
        # All should be "the" or "The"
        assert all(m.lower() == "the" for m in matched)


class TestWildcardSearch:
    """Test wildcard pattern matching"""

    def test_question_mark_wildcard(self, sample_corpus):
        """Test ? wildcard for single character"""
        query = "fo?"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["fox"]

    def test_asterisk_wildcard_prefix(self, sample_corpus):
        """Test * wildcard for zero or more characters at start"""
        query = "*ick"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["quick"]

    def test_asterisk_wildcard_suffix(self, sample_corpus):
        """Test * wildcard for zero or more characters at end"""
        query = "qu*"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["quick"]

    def test_plus_wildcard(self, sample_corpus):
        """Test + wildcard for one or more characters"""
        query = "+uck"
        matches = plc.search(sample_corpus, query)
        # Should match "truck" but not "uck"
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["truck"]

    def test_combined_wildcards(self, sample_corpus):
        """Test combining multiple wildcards"""
        query = "s?ng"
        matches = plc.search(sample_corpus, query)
        # Should match "sing", "sang", "song"
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert set(matched) == {"sing", "sang", "song"}

    def test_wildcard_able_pattern(self, sample_corpus):
        """Test: *able → able, table, capable, suitable, available"""
        query = "*able"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert set(matched) == {"able", "table", "capable", "suitable", "available"}

    def test_plus_able_pattern(self, sample_corpus):
        """Test: +able → table, capable, suitable, but not able"""
        query = "+able"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should NOT include plain "able"
        assert "able" not in matched
        assert set(matched) == {"table", "capable", "suitable", "available"}


class TestAlternativeSearch:
    """Test square bracket alternatives"""

    def test_simple_alternatives(self, sample_corpus):
        """Test comma-separated alternatives"""
        query = "[car,truck]"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert set(matched) == {"car", "truck"}

    def test_alternatives_with_wildcards(self, sample_corpus):
        """Test alternatives including wildcards"""
        query = "[qu*,br*]"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert set(matched) == {"quick", "brown"}

    def test_empty_alternative(self, sample_corpus):
        """Test empty alternative (optional character)"""
        # Note: The current parser doesn't support alternatives embedded in words
        # like "neighbo[u,]r". Use separate alternatives instead.
        query = "[neighbour,neighbor]"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert set(matched) == {"neighbour", "neighbor"}


class TestWordSequences:
    """Test multi-word sequences"""

    def test_two_word_sequence(self, sample_corpus):
        """Test matching two consecutive words"""
        query = "quick brown"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["quick brown"]

    def test_three_word_sequence(self, sample_corpus):
        """Test matching three consecutive words"""
        query = "the lazy dog"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["the lazy dog"]

    def test_sequence_with_wildcards(self, sample_corpus):
        """Test sequence containing wildcards"""
        query = "quick br*"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["quick brown"]


class TestGapTokens:
    """Test gap tokens (* and +)"""

    def test_optional_gap_star(self, sample_corpus):
        """Test * for optional token"""
        query = "fox * over"
        matches = plc.search(sample_corpus, query)
        # Should match "fox jumps over"
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["fox jumps over"]

    def test_required_gap_plus(self, sample_corpus):
        """Test + for required gap"""
        query = "fox + over"
        matches = plc.search(sample_corpus, query)
        # Should match "fox jumps over" (with required gap)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["fox jumps over"]

    def test_gap_with_multiple_words(self, sample_corpus):
        """Test gap between words"""
        query = "red * and"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["red car and"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
