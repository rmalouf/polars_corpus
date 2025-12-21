"""Tests for lexical richness measures."""

import polars as pl
import polars_corpus as plc
import pytest


class TestTTR:
    """Test type-token ratio calculation."""

    def test_basic_ttr(self):
        """Test TTR with simple data."""
        df = pl.DataFrame({"words": ["the", "cat", "sat", "on", "the", "mat"]})
        result = df.select(plc.ttr("words")).item()
        # 5 unique words out of 6 total = 5/6 ≈ 0.833
        assert result == pytest.approx(5 / 6)

    def test_ttr_all_unique(self):
        """Test TTR when all tokens are unique."""
        df = pl.DataFrame({"words": ["a", "b", "c", "d", "e"]})
        result = df.select(plc.ttr("words")).item()
        assert result == 1.0

    def test_ttr_all_same(self):
        """Test TTR when all tokens are identical."""
        df = pl.DataFrame({"words": ["the"] * 10})
        result = df.select(plc.ttr("words")).item()
        assert result == 0.1

    def test_ttr_empty(self):
        """Test TTR with empty data."""
        df = pl.DataFrame({"words": []}, schema={"words": pl.String})
        result = df.select(plc.ttr("words")).item()
        # 0/0 = NaN
        assert result != result  # NaN != NaN

    def test_ttr_with_nulls(self):
        """Test TTR treats null as a distinct type (Polars default)."""
        # Nulls should count as a unique type
        df = pl.DataFrame({"words": ["a", "b", None, "a", "b", None]})
        result = df.select(plc.ttr("words")).item()
        # 3 unique values (a, b, None) out of 6 total = 0.5
        assert result == pytest.approx(0.5)


class TestMSTTR:
    """Test mean segmental type-token ratio calculation."""

    def test_basic_msttr(self):
        """Test MSTTR with known values."""
        # Create 3 complete segments of 10 tokens each
        # Segment 1: all unique (TTR = 1.0)
        # Segment 2: all same (TTR = 0.1)
        # Segment 3: half unique (TTR = 0.5)
        tokens = (
            ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]  # 10 unique
            + ["x"] * 10  # 1 unique
            + ["p", "q", "r", "s", "t", "p", "q", "r", "s", "t"]  # 5 unique
        )
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=10)).item()
        expected = (1.0 + 0.1 + 0.5) / 3
        assert result == pytest.approx(expected)

    def test_msttr_default_n(self):
        """Test MSTTR with default segment size."""
        # Create 2 complete segments of 1000 tokens
        tokens = list(range(500)) * 4  # 500 unique repeated, 2000 total
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens")).item()
        # Each 1000-token segment has 500 unique tokens
        assert result == pytest.approx(0.5)

    def test_msttr_drops_incomplete_segment(self):
        """Test that incomplete final segment is dropped."""
        # 25 tokens with n=10 should give 2 complete segments
        tokens = ["a"] * 10 + ["b"] * 10 + ["c"] * 5
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=10)).item()
        # Only first two segments: (0.1 + 0.1) / 2 = 0.1
        assert result == pytest.approx(0.1)

    def test_msttr_single_segment(self):
        """Test MSTTR with exactly one complete segment."""
        tokens = list("abcdefghij")  # 10 unique tokens
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=10)).item()
        assert result == 1.0

    def test_msttr_no_complete_segments(self):
        """Test MSTTR when there are no complete segments."""
        tokens = ["a", "b", "c"]  # Only 3 tokens with n=10
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=10)).item()
        assert result is None

    def test_msttr_empty(self):
        """Test MSTTR with empty data."""
        df = pl.DataFrame({"tokens": []}, schema={"tokens": pl.String})
        result = df.select(plc.msttr("tokens", n=10)).item()
        assert result is None

    def test_msttr_various_types(self):
        """Test MSTTR works with different data types."""
        # Should work with integers, strings, etc.
        df = pl.DataFrame({"nums": [1, 2, 3, 4, 5] * 4})
        result = df.select(plc.msttr("nums", n=10)).item()
        # Each segment of 10 has 5 unique values
        assert result == pytest.approx(0.5)

    def test_msttr_segment_size_1(self):
        """Test MSTTR with segment size of 1."""
        tokens = ["a", "b", "c", "d", "e"]
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=1)).item()
        # Each segment has 1 token, all unique, so TTR = 1.0 for each
        assert result == 1.0

    def test_msttr_multiple_segments_varying_diversity(self):
        """Test MSTTR correctly averages across segments with varying diversity."""
        # 4 segments of 5 tokens each
        # Segment 1: TTR = 1.0 (all unique)
        # Segment 2: TTR = 0.2 (1 unique)
        # Segment 3: TTR = 0.6 (3 unique)
        # Segment 4: TTR = 0.4 (2 unique)
        tokens = (
            ["a", "b", "c", "d", "e"]
            + ["x", "x", "x", "x", "x"]
            + ["p", "q", "r", "p", "q"]
            + ["m", "n", "m", "n", "m"]
        )
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=5)).item()
        expected = (1.0 + 0.2 + 0.6 + 0.4) / 4
        assert result == pytest.approx(expected)

    def test_msttr_with_nulls(self):
        """Test MSTTR handles null values appropriately."""
        # Polars n_unique() counts null as a distinct value
        tokens = ["a", "b", None, "a", "b"] * 4
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.msttr("tokens", n=10)).item()
        # Each segment has "a", "b", None appearing - 3 unique out of 10
        assert result == pytest.approx(0.3)

    def test_msttr_invalid_n(self):
        """Test MSTTR raises error for invalid n values."""
        with pytest.raises(ValueError, match="must be greater than 0"):
            plc.msttr("tokens", n=0)

        with pytest.raises(ValueError, match="must be greater than 0"):
            plc.msttr("tokens", n=-10)

        with pytest.raises(TypeError, match="must be an integer"):
            plc.msttr("tokens", n=10.5)


class TestMTLD:
    """Test Measure of Textual Lexical Diversity calculation."""

    def test_mtld_basic(self):
        """Test MTLD with varying diversity levels."""
        # High diversity (all unique) = high MTLD
        df_high = pl.DataFrame({"tokens": [f"word{i}" for i in range(100)]})
        assert df_high.select(plc.mtld("tokens")).item() > 50

        # Low diversity (repetitive) = low MTLD
        df_low = pl.DataFrame({"tokens": ["the"] * 100})
        assert df_low.select(plc.mtld("tokens")).item() < 10

    def test_mtld_minimum_tokens(self):
        """Test MTLD requires at least 10 tokens."""
        df_valid = pl.DataFrame({"tokens": [f"w{i}" for i in range(10)]})
        assert df_valid.select(plc.mtld("tokens")).item() is not None

        df_invalid = pl.DataFrame({"tokens": [f"w{i}" for i in range(9)]})
        assert df_invalid.select(plc.mtld("tokens")).item() is None

    def test_mtld_with_nulls(self):
        """Test MTLD treats null as a distinct token (Polars default)."""
        tokens = [f"word{i}" for i in range(10)] + [None, None]
        df = pl.DataFrame({"tokens": tokens})
        result = df.select(plc.mtld("tokens")).item()
        assert result is not None and result > 0

    def test_mtld_custom_threshold(self):
        """Test MTLD accepts custom threshold parameter."""
        df = pl.DataFrame({"tokens": ["the"] * 100})
        result_default = df.select(plc.mtld("tokens")).item()
        result_custom = df.select(plc.mtld("tokens", threshold=0.800)).item()
        assert result_default > 0
        assert result_custom > 0

    def test_mtld_invalid_threshold(self):
        """Test MTLD raises error for invalid threshold values."""
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plc.mtld("tokens", threshold=0.0)

        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plc.mtld("tokens", threshold=1.0)

        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plc.mtld("tokens", threshold=-0.5)

        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plc.mtld("tokens", threshold=1.5)


class TestYulesK:
    """Test Yule's K characteristic calculation."""

    def test_yules_k_basic(self):
        """Test Yule's K with varying diversity levels."""
        # All unique = K = 0
        df_unique = pl.DataFrame({"tokens": list("abcdefghij")})
        assert df_unique.select(plc.yules_k("tokens")).item() == pytest.approx(0.0)

        # Repetitive = high K (low diversity)
        df_repetitive = pl.DataFrame({"tokens": ["the"] * 50 + ["cat"] * 30})
        assert df_repetitive.select(plc.yules_k("tokens")).item() > 100

    def test_yules_k_with_nulls(self):
        """Test Yule's K treats null as a distinct type (Polars default)."""
        df = pl.DataFrame({"tokens": ["a", "b", None, "a", "b", None]})
        assert df.select(plc.yules_k("tokens")).item() > 0
