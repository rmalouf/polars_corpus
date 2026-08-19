"""Tests for lexical richness measures."""

import polars as pl
import polars_corpus as plc
import pytest


@pytest.mark.parametrize(
    "tokens,expected",
    [
        pytest.param(["the", "cat", "sat", "on", "the", "mat"], 5 / 6, id="mixed"),
        pytest.param(list("abcde"), 1.0, id="all-unique"),
        pytest.param(["the"] * 10, 0.1, id="all-identical"),
        # Polars counts null as a distinct type: a, b, None out of 6 tokens.
        pytest.param(["a", "b", None, "a", "b", None], 0.5, id="nulls-are-a-type"),
    ],
)
def test_ttr(tokens, expected):
    df = pl.DataFrame({"words": tokens}, schema={"words": pl.String})
    assert df.select(plc.ttr("words")).item() == pytest.approx(expected)


def test_ttr_empty_is_nan():
    df = pl.DataFrame({"words": []}, schema={"words": pl.String})
    result = df.select(plc.ttr("words")).item()
    assert result != result  # 0/0 is NaN


class TestMSTTR:
    """Mean segmental type-token ratio: TTR averaged over fixed-size segments."""

    @pytest.mark.parametrize(
        "tokens,n,expected",
        [
            pytest.param(
                list("abcdefghij")  # segment TTR 1.0
                + ["x"] * 10  # segment TTR 0.1
                + list("pqrstpqrst"),  # segment TTR 0.5
                10,
                (1.0 + 0.1 + 0.5) / 3,
                id="three-segments",
            ),
            pytest.param(
                list("abcde")  # 1.0
                + ["x"] * 5  # 0.2
                + list("pqrpq")  # 0.6
                + list("mnmnm"),  # 0.4
                5,
                (1.0 + 0.2 + 0.6 + 0.4) / 4,
                id="varying-diversity",
            ),
            pytest.param(list("abcdefghij"), 10, 1.0, id="exactly-one-segment"),
            pytest.param(list("abcde"), 1, 1.0, id="segment-size-1"),
            # 25 tokens at n=10 yields two complete segments; the tail is dropped.
            pytest.param(["a"] * 10 + ["b"] * 10 + ["c"] * 5, 10, 0.1, id="drops-tail"),
            # Null counts as a type: a, b, None appear in each 10-token segment.
            pytest.param(
                ["a", "b", None, "a", "b"] * 4, 10, 0.3, id="nulls-are-a-type"
            ),
        ],
    )
    def test_msttr(self, tokens, n, expected):
        df = pl.DataFrame({"tokens": tokens}, schema={"tokens": pl.String})
        assert df.select(plc.msttr("tokens", n=n)).item() == pytest.approx(expected)

    def test_default_segment_size(self):
        """The default n is 1000."""
        df = pl.DataFrame({"tokens": list(range(500)) * 4})  # 2000 tokens, 500 types
        assert df.select(plc.msttr("tokens")).item() == pytest.approx(0.5)

    def test_non_string_tokens(self):
        df = pl.DataFrame({"nums": [1, 2, 3, 4, 5] * 4})
        assert df.select(plc.msttr("nums", n=10)).item() == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "tokens",
        [
            pytest.param(["a", "b", "c"], id="fewer-tokens-than-n"),
            pytest.param([], id="empty"),
        ],
    )
    def test_no_complete_segments_is_null(self, tokens):
        df = pl.DataFrame({"tokens": tokens}, schema={"tokens": pl.String})
        assert df.select(plc.msttr("tokens", n=10)).item() is None

    @pytest.mark.parametrize(
        "n,exc,message",
        [
            (0, ValueError, "must be greater than 0"),
            (-10, ValueError, "must be greater than 0"),
            (10.5, TypeError, "must be an integer"),
        ],
    )
    def test_invalid_n(self, n, exc, message):
        with pytest.raises(exc, match=message):
            plc.msttr("tokens", n=n)


class TestMTLD:
    """Measure of Textual Lexical Diversity."""

    @pytest.mark.parametrize(
        "tokens,predicate",
        [
            pytest.param(
                [f"word{i}" for i in range(100)], lambda v: v > 50, id="high-diversity"
            ),
            pytest.param(["the"] * 100, lambda v: v < 10, id="low-diversity"),
        ],
    )
    def test_tracks_diversity(self, tokens, predicate):
        df = pl.DataFrame({"tokens": tokens})
        assert predicate(df.select(plc.mtld("tokens")).item())

    @pytest.mark.parametrize(
        "n_tokens,defined",
        [(10, True), (9, False)],
        ids=["at-minimum", "below-minimum"],
    )
    def test_requires_ten_tokens(self, n_tokens, defined):
        df = pl.DataFrame({"tokens": [f"w{i}" for i in range(n_tokens)]})
        assert (df.select(plc.mtld("tokens")).item() is not None) is defined

    def test_nulls_are_a_token(self):
        tokens = [f"word{i}" for i in range(10)] + [None, None]
        df = pl.DataFrame({"tokens": tokens})
        assert df.select(plc.mtld("tokens")).item() > 0

    def test_custom_threshold(self):
        df = pl.DataFrame({"tokens": ["the"] * 100})
        assert df.select(plc.mtld("tokens", threshold=0.800)).item() > 0

    @pytest.mark.parametrize("threshold", [0.0, 1.0, -0.5, 1.5])
    def test_invalid_threshold(self, threshold):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            plc.mtld("tokens", threshold=threshold)


@pytest.mark.parametrize(
    "tokens,predicate",
    [
        pytest.param(
            list("abcdefghij"), lambda v: v == pytest.approx(0.0), id="all-unique"
        ),
        pytest.param(["the"] * 50 + ["cat"] * 30, lambda v: v > 100, id="repetitive"),
        pytest.param(
            ["a", "b", None, "a", "b", None], lambda v: v > 0, id="with-nulls"
        ),
    ],
)
def test_yules_k(tokens, predicate):
    """Yule's K is 0 for maximal diversity and grows as repetition increases."""
    df = pl.DataFrame({"tokens": tokens}, schema={"tokens": pl.String})
    assert predicate(df.select(plc.yules_k("tokens")).item())


class TestVocabularyGrowth:
    """Vocabulary growth curve: a running count of types."""

    @pytest.mark.parametrize(
        "tokens,expected",
        [
            pytest.param(list("abcde"), [1, 2, 3, 4, 5], id="all-unique"),
            pytest.param(["the"] * 4, [1, 1, 1, 1], id="all-identical"),
            pytest.param(
                ["the", "cat", "sat", "on", "the", "mat"],
                [1, 2, 3, 4, 4, 5],
                id="mixed",
            ),
            # Null is a type, as in ttr, and only its first occurrence counts.
            pytest.param(["a", None, "a", None, "b"], [1, 2, 2, 2, 3], id="nulls"),
            pytest.param([], [], id="empty"),
        ],
    )
    def test_growth(self, tokens, expected):
        df = pl.DataFrame({"tokens": tokens}, schema={"tokens": pl.String})
        curve = df.select(plc.vocabulary_growth("tokens")).to_series()
        assert curve.to_list() == expected

    @pytest.mark.parametrize("dtype", [pl.String, pl.Categorical, pl.Int32], ids=str)
    def test_dtypes(self, dtype):
        tokens = pl.Series(["1", "2", "1", "3"]).cast(dtype)
        df = pl.DataFrame({"tokens": tokens})
        assert df.select(plc.vocabulary_growth("tokens")).to_series().to_list() == [
            1,
            2,
            2,
            3,
        ]

    def test_ends_at_n_unique(self):
        df = pl.DataFrame({"tokens": ["a", "b", "c", "a", "b"] * 20})
        last = df.select(plc.vocabulary_growth("tokens")).to_series()[-1]
        assert last == df.select(pl.col("tokens").n_unique()).item()

    def test_per_file(self):
        """The curve restarts for each group when used with `over`."""
        df = pl.DataFrame(
            {
                "file_id": ["a", "a", "a", "b", "b", "b"],
                "tokens": ["x", "y", "x", "x", "x", "z"],
            }
        )
        curve = df.select(plc.vocabulary_growth("tokens").over("file_id")).to_series()
        assert curve.to_list() == [1, 2, 2, 1, 1, 2]
