import polars as pl
import pytest

from polars_corpus._typing import Span
from polars_corpus.spans import with_span_index, with_spans


class TestWithSpanIndex:
    """Tests for with_span_index function."""

    def test_basic_bio_tagging(self):
        """Test basic BIO sequence produces correct span indices."""
        df = pl.DataFrame(
            {
                "token": ["The", "quick", "brown", "fox", "jumps"],
                "bio": ["B", "I", "O", "B", "I"],
            }
        )

        result = with_span_index(df, "bio")
        expected_indices = [1, 1, None, 2, 2]

        assert result["span_idx"].to_list() == expected_indices

    def test_single_token_spans(self):
        """Test spans that are only one token long."""
        df = pl.DataFrame(
            {"token": ["The", "quick", "brown", "fox"], "bio": ["B", "O", "B", "O"]}
        )

        result = with_span_index(df, "bio")
        expected_indices = [1, None, 2, None]

        assert result["span_idx"].to_list() == expected_indices

    def test_consecutive_spans(self):
        """Test multiple consecutive spans."""
        df = pl.DataFrame(
            {
                "token": ["A", "B", "C", "D", "E", "F"],
                "bio": ["B", "I", "B", "I", "I", "O"],
            }
        )

        result = with_span_index(df, "bio")
        expected_indices = [1, 1, 2, 2, 2, None]

        assert result["span_idx"].to_list() == expected_indices

    def test_all_outside_spans(self):
        """Test sequence with all O tags."""
        df = pl.DataFrame({"token": ["The", "quick", "brown"], "bio": ["O", "O", "O"]})

        result = with_span_index(df, "bio")
        expected_indices = [None, None, None]

        assert result["span_idx"].to_list() == expected_indices

    def test_all_inside_one_span(self):
        """Test sequence that is entirely one span."""
        df = pl.DataFrame({"token": ["New", "York", "City"], "bio": ["B", "I", "I"]})

        result = with_span_index(df, "bio")
        expected_indices = [1, 1, 1]

        assert result["span_idx"].to_list() == expected_indices

    def test_custom_column_name(self):
        """Test using custom column name for span index."""
        df = pl.DataFrame({"token": ["The", "quick"], "bio": ["B", "I"]})

        result = with_span_index(df, "bio", name="my_span_idx")

        assert "my_span_idx" in result.columns
        assert result["my_span_idx"].to_list() == [1, 1]

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pl.DataFrame(
            {"token": [], "bio": []}, schema={"token": pl.Utf8, "bio": pl.Utf8}
        )

        result = with_span_index(df, "bio")

        assert len(result) == 0
        assert "span_idx" in result.columns

    def test_invalid_scheme_raises_error(self):
        """Test that non-BIO schemes raise NotImplementedError."""
        df = pl.DataFrame({"token": ["The"], "bio": ["B"]})

        with pytest.raises(NotImplementedError, match="Only BIO is supported"):
            with_span_index(df, "bio", scheme="BILOU")

    def test_malformed_bio_sequence(self):
        """Test behavior with malformed BIO sequences (I without B)."""
        df = pl.DataFrame(
            {
                "token": ["The", "quick", "brown"],
                "bio": ["I", "I", "O"],  # Starts with I instead of B
            }
        )

        # This should still work - the function should handle it gracefully
        # I tags without preceding B should get index 0 (since cum_sum starts at 0)
        result = with_span_index(df, "bio")
        expected_indices = [0, 0, None]  # or whatever the expected behavior is

        assert result["span_idx"].to_list() == expected_indices


class TestWithSpans:
    """Tests for with_spans function."""

    def test_single_span(self):
        """Test adding a single span to dataframe."""
        df = pl.DataFrame(
            {
                "token": ["The", "quick", "brown", "fox", "jumps"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB"],
            }
        )

        concordance = [Span(1, 3)]  # "quick brown"
        result = with_spans(df, concordance)
        expected_spans = ["O", "B", "I", "O", "O"]

        assert result["spans"].to_list() == expected_spans

    def test_multiple_non_overlapping_spans(self):
        """Test multiple non-overlapping spans."""
        df = pl.DataFrame(
            {
                "token": ["The", "quick", "brown", "fox", "jumps", "high"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADV"],
            }
        )

        concordance = [Span(1, 3), Span(4, 6)]  # "quick brown" and "jumps high"
        result = with_spans(df, concordance)
        expected_spans = ["O", "B", "I", "O", "B", "I"]

        assert result["spans"].to_list() == expected_spans

    def test_single_token_spans(self):
        """Test spans that cover only one token."""
        df = pl.DataFrame(
            {
                "token": ["The", "quick", "brown", "fox"],
                "pos": ["DET", "ADJ", "ADJ", "NOUN"],
            }
        )

        concordance = [Span(0, 1), Span(3, 4)]  # "The" and "fox"
        result = with_spans(df, concordance)
        expected_spans = ["B", "O", "O", "B"]

        assert result["spans"].to_list() == expected_spans

    def test_adjacent_spans(self):
        """Test adjacent spans (end of one = start of next)."""
        df = pl.DataFrame(
            {
                "token": ["New", "York", "City", "Mayor"],
                "pos": ["PROPN", "PROPN", "PROPN", "NOUN"],
            }
        )

        concordance = [Span(0, 2), Span(2, 4)]  # "New York" and "City Mayor"
        result = with_spans(df, concordance)
        expected_spans = ["B", "I", "B", "I"]

        assert result["spans"].to_list() == expected_spans

    def test_span_validation_helper(self):
        """Helper test to validate that concordances don't contain overlaps.

        This could be useful as a utility function in your module.
        """

        def has_overlapping_spans(concordance):
            """Check if concordance contains overlapping spans."""
            sorted_spans = sorted(concordance, key=lambda x: x.start)
            for i in range(len(sorted_spans) - 1):
                if sorted_spans[i].end > sorted_spans[i + 1].start:
                    return True
            return False

        # Valid non-overlapping spans
        valid_concordance = [Span(0, 2), Span(3, 5), Span(7, 9)]
        assert not has_overlapping_spans(valid_concordance)

        # Invalid overlapping spans
        invalid_concordance = [Span(0, 3), Span(2, 5)]  # overlap at position 2
        assert has_overlapping_spans(invalid_concordance)

        # Adjacent spans (touching but not overlapping) - should be valid
        adjacent_concordance = [Span(0, 2), Span(2, 4)]
        assert not has_overlapping_spans(adjacent_concordance)

    def test_empty_concordance(self):
        """Test with empty concordance list."""
        df = pl.DataFrame(
            {"token": ["The", "quick", "brown"], "pos": ["DET", "ADJ", "ADJ"]}
        )

        concordance = []
        result = with_spans(df, concordance)
        expected_spans = ["O", "O", "O"]

        assert result["spans"].to_list() == expected_spans

    def test_span_covering_entire_dataframe(self):
        """Test span that covers the entire dataframe."""
        df = pl.DataFrame(
            {"token": ["All", "tokens", "covered"], "pos": ["DET", "NOUN", "VERB"]}
        )

        concordance = [Span(0, 3)]
        result = with_spans(df, concordance)
        expected_spans = ["B", "I", "I"]

        assert result["spans"].to_list() == expected_spans

    def test_custom_column_name(self):
        """Test using custom column name for spans."""
        df = pl.DataFrame({"token": ["The", "quick"], "pos": ["DET", "ADJ"]})

        concordance = [Span(0, 2)]
        result = with_spans(df, concordance, name="my_spans")

        assert "my_spans" in result.columns
        assert result["my_spans"].to_list() == ["B", "I"]

    def test_out_of_bounds_spans(self):
        """Test spans that extend beyond dataframe boundaries."""
        df = pl.DataFrame({"token": ["The", "quick"], "pos": ["DET", "ADJ"]})

        concordance = [Span(0, 5)]  # Extends beyond dataframe

        # This should either truncate gracefully or raise an appropriate error
        with pytest.raises((IndexError, ValueError)):
            with_spans(df, concordance)

    def test_negative_span_positions(self):
        """Test spans with negative positions."""
        df = pl.DataFrame({"token": ["The", "quick"], "pos": ["DET", "ADJ"]})

        concordance = [Span(-1, 1)]  # Negative start

        # Should handle gracefully or raise appropriate error
        with pytest.raises((IndexError, ValueError)):
            with_spans(df, concordance)

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pl.DataFrame(
            {"token": [], "pos": []}, schema={"token": pl.Utf8, "pos": pl.Utf8}
        )

        concordance = []
        result = with_spans(df, concordance)

        assert len(result) == 0
        assert "spans" in result.columns


class TestIntegration:
    """Integration tests using both functions together."""

    def test_spans_to_index_roundtrip(self):
        """Test that spans can be converted to BIO and then to indices correctly."""
        df = pl.DataFrame(
            {
                "token": ["The", "New", "York", "Times", "reported"],
                "pos": ["DET", "PROPN", "PROPN", "PROPN", "VERB"],
            }
        )

        concordance = [Span(1, 4)]  # "New York Times"

        # Add spans
        df_with_spans = with_spans(df, concordance)

        # Convert to span indices
        df_with_indices = with_span_index(df_with_spans, "spans")

        expected_spans = ["O", "B", "I", "I", "O"]
        expected_indices = [None, 1, 1, 1, None]

        assert df_with_indices["spans"].to_list() == expected_spans
        assert df_with_indices["span_idx"].to_list() == expected_indices

    def test_multiple_spans_to_indices(self):
        """Test multiple spans converted to indices."""
        df = pl.DataFrame(
            {
                "token": ["John", "Smith", "met", "Mary", "Johnson", "yesterday"],
                "pos": ["PROPN", "PROPN", "VERB", "PROPN", "PROPN", "ADV"],
            }
        )

        concordance = [Span(0, 2), Span(3, 5)]  # "John Smith" and "Mary Johnson"

        df_with_spans = with_spans(df, concordance)
        df_with_indices = with_span_index(df_with_spans, "spans")

        expected_spans = ["B", "I", "O", "B", "I", "O"]
        expected_indices = [1, 1, None, 2, 2, None]

        assert df_with_indices["spans"].to_list() == expected_spans
        assert df_with_indices["span_idx"].to_list() == expected_indices


# Fixtures for more complex testing scenarios
@pytest.fixture
def sample_corpus_df():
    """Sample corpus dataframe for testing."""
    return pl.DataFrame(
        {
            "token": [
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
            "pos": ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN"],
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
def complex_concordance():
    """Complex concordance with multiple spans for testing."""
    return [
        Span(1, 4),  # "quick brown fox"
        Span(6, 9),  # "the lazy dog"
    ]


class TestCorpusLinguisticsScenarios:
    """Tests for common corpus linguistics scenarios."""

    def test_noun_phrase_extraction(self, sample_corpus_df):
        """Test extracting noun phrases."""
        # Simulate noun phrase spans
        np_concordance = [Span(1, 4), Span(6, 9)]  # Adjective + Noun phrases

        result = with_spans(sample_corpus_df, np_concordance, name="np_spans")
        result = with_span_index(result, "np_spans", name="np_idx")

        # Check that noun phrases are correctly tagged
        np_tokens = result.filter(pl.col("np_spans") != "O")["token"].to_list()
        expected_np_tokens = ["quick", "brown", "fox", "the", "lazy", "dog"]

        assert np_tokens == expected_np_tokens

    def test_multiple_annotation_layers(self, sample_corpus_df):
        """Test multiple non-overlapping annotation layers (e.g., syntactic vs semantic spans)."""
        # Non-overlapping syntactic and semantic spans
        syntactic_spans = [Span(1, 4)]  # "quick brown fox"
        semantic_spans = [Span(6, 9)]  # "the lazy dog" (different span)

        result = with_spans(sample_corpus_df, syntactic_spans, name="syntax")
        result = with_spans(result, semantic_spans, name="semantics")

        # Both span types should be present
        assert "syntax" in result.columns
        assert "semantics" in result.columns

        # Verify non-overlapping nature
        syntax_spans = result["syntax"].to_list()
        semantic_spans = result["semantics"].to_list()

        expected_syntax = ["O", "B", "I", "I", "O", "O", "O", "O", "O"]
        expected_semantics = ["O", "O", "O", "O", "O", "O", "B", "I", "I"]

        assert syntax_spans == expected_syntax
        assert semantic_spans == expected_semantics
