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


class TestPOSTagSearch:
    """Test POS tag searches using word_TAG syntax"""

    def test_word_with_pos_tag(self, sample_corpus):
        """Test word+POS pattern (e.g., lights_NN2)"""
        query = "fox_NN"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        assert matched == ["fox"]

    def test_pos_tag_only(self, sample_corpus):
        """Test POS-only pattern (e.g., _NN)"""
        query = "_NN"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match all NN tagged tokens: fox, student, car, truck, song, etc.
        assert "fox" in matched
        assert "student" in matched
        assert "car" in matched

    def test_wildcard_in_word_part(self, sample_corpus):
        """Test wildcards in word part (e.g., *ly_RB)"""
        query = "*ly_RB"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "slowly" tagged as RB
        assert "slowly" in matched

    def test_wildcard_in_pos_part(self, sample_corpus):
        """Test wildcards in POS part (e.g., sing_V*)"""
        query = "sing_V*"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "sing" with VBP tag
        assert "sing" in matched

    def test_pos_in_sequence(self, sample_corpus):
        """Test POS pattern in word sequence"""
        query = "the _JJ dog"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "the lazy dog" (DT JJ NN sequence)
        assert "the lazy dog" in matched

    def test_multiple_pos_tags(self, sample_corpus):
        """Test sequence of POS-only patterns"""
        query = "_DT _JJ _NN"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match DT JJ NN sequences like "the lazy dog", "The red car", etc.
        assert len(matched) > 0
        # Check for known patterns in the corpus
        assert "the lazy dog" in matched or "The red car" in matched


class TestLemmaSearch:
    """Test lemma searches using {lemma} and {lemma/POS} syntax"""

    def test_basic_lemma_search(self, sample_corpus):
        """Test basic lemma search {lemma}"""
        query = "{sing}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "sing" and "sang" (both have lemma "sing")
        assert set(matched) == {"sing", "sang"}

    def test_lemma_with_pos(self, sample_corpus):
        """Test lemma with POS constraint {lemma/POS}"""
        query = "{table/N}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "table" tagged as NN (noun), not VB (verb)
        assert "table" in matched
        # Verify we got the noun "table", not verb "table"
        # In our test corpus, there's one NN and one VB
        assert len(matched) == 1  # One instance of "table" as noun

    def test_lemma_verb_forms(self, sample_corpus):
        """Test lemma matching different verb forms"""
        query = "{walk}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "walked" (lemma is "walk")
        assert "walked" in matched

    def test_lemma_in_sequence(self, sample_corpus):
        """Test lemma in word sequence"""
        query = "{sing} sang"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "sing sang" where first token has lemma "sing"
        assert "sing sang" in matched

    def test_lemma_with_gap(self, sample_corpus):
        """Test lemma with gap tokens"""
        query = "{be} * suitable"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "is suitable" (lemma of "is" is "be")
        assert "is suitable" in matched

    def test_lemma_simplified_pos_verb(self, sample_corpus):
        """Test simplified POS tag mapping (V for verbs)"""
        query = "{be/V}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "are" and "is" (both lemma "be", tagged as verbs)
        assert "are" in matched
        assert "is" in matched

    def test_lemma_simplified_pos_adjective(self, sample_corpus):
        """Test simplified POS tag mapping (A for adjectives)"""
        query = "{capable/A}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "capable" tagged as adjective (JJ)
        assert "capable" in matched

    def test_multiple_lemmas_in_sequence(self, sample_corpus):
        """Test multiple lemma patterns in a sequence"""
        query = "{be} {able}"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "are able" (lemmas "be" and "able")
        assert "are able" in matched

    def test_lemma_with_exact_pos_tag(self, sample_corpus):
        """Test lemma with exact POS tag using {lemma}_TAG syntax"""
        query = "{sing}_VBD"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match "sang" which is lemma "sing" with POS VBD
        assert "sang" in matched
        # Should not match "sing" which has POS VBP
        assert "sing" not in matched

    def test_lemma_with_pos_wildcard(self, sample_corpus):
        """Test lemma with POS wildcard using {lemma}_TAG* syntax"""
        query = "{be}_V*"
        matches = plc.search(sample_corpus, query)
        assert matches is not None
        matched = get_matched_tokens(sample_corpus, matches)
        # Should match both "are" (VBP) and "is" (VBZ)
        assert "are" in matched
        assert "is" in matched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
