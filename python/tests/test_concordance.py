"""Tests for concordancing: context windows, collocates, and result slicing."""

import polars as pl
import pytest
from polars_corpus import Match, SearchResults, Span

from .helpers import corpus

# "the cat sat on the mat . the dog sat on the log ."
TOKENS = "the cat sat on the mat . the dog sat on the log ."


@pytest.fixture
def results():
    """The two "sat" matches, at positions 2 and 9."""
    df = corpus(token=TOKENS)
    return SearchResults(df, "sat", [Match(Span(2, 3), {}), Match(Span(9, 10), {})])


class TestWindow:
    """The context a window asks for, and what the corpus can give it"""

    def test_context_columns(self, results):
        conc = results.concordance("token", window=2)

        assert conc.columns == [
            "token_left_context",
            "token",
            "token_right_context",
        ]
        assert conc["token_left_context"].to_list() == [
            ["the", "cat"],
            ["the", "dog"],
        ]
        assert conc["token"].to_list() == [["sat"], ["sat"]]
        assert conc["token_right_context"].to_list() == [["on", "the"], ["on", "the"]]

    def test_no_context_by_default(self, results):
        conc = results.concordance("token")

        assert conc.columns == ["token"]
        assert conc["token"].to_list() == [["sat"], ["sat"]]

    def test_window_zero_is_the_default(self, results):
        assert results.concordance("token", window=0).columns == ["token"]

    def test_window_truncated_at_the_corpus_edges(self):
        df = corpus(token="the cat sat")
        results = SearchResults(df, "", [Match(Span(0, 1), {}), Match(Span(2, 3), {})])
        conc = results.concordance("token", window=5)

        # Neither context is padded out to the five tokens asked for.
        assert conc["token_left_context"].to_list() == [[], ["the", "cat"]]
        assert conc["token_right_context"].to_list() == [["cat", "sat"], []]

    def test_context_runs_across_files(self):
        """A window is not stopped by a change of file id -- see concordance()."""
        df = pl.DataFrame(
            {"token": ["a", "b", "c", "d"], "file_id": ["1", "1", "2", "2"]}
        )
        results = SearchResults(df, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", window=2)

        assert conc["token_left_context"].to_list() == [["a", "b"]]

    @pytest.mark.parametrize("window", [-1, 1.5, None, "2", True])
    def test_bad_window(self, results, window):
        with pytest.raises(ValueError, match="window must be a non-negative integer"):
            results.concordance("token", window=window)

    def test_several_columns(self):
        df = corpus(token="the cat sat", pos="DT NN VB")
        results = SearchResults(df, "", [Match(Span(1, 2), {})])
        conc = results.concordance(["token", "pos"], window=1)

        assert conc.columns == [
            "token_left_context",
            "token",
            "token_right_context",
            "pos_left_context",
            "pos",
            "pos_right_context",
        ]
        assert conc["pos"].to_list() == [["NN"]]


class TestChunkColumn:
    """Context taken out to the chunk boundaries rather than a fixed window"""

    @pytest.fixture
    def chunked(self):
        # Two sentences: "the cat sat ." and "the dog sat ."
        df = corpus(token="the cat sat . the dog sat .", chunks="B I I I B I I I")
        return df

    def test_context_reaches_the_chunk_edges(self, chunked):
        results = SearchResults(chunked, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["the", "cat"]]
        assert conc["token_right_context"].to_list() == [["."]]

    def test_context_stops_at_the_next_chunk(self, chunked):
        results = SearchResults(chunked, "", [Match(Span(6, 7), {})])
        conc = results.concordance("token", chunk_column="chunks")

        # The second sentence only, not back over the first.
        assert conc["token_left_context"].to_list() == [["the", "dog"]]

    def test_match_at_the_start_of_the_corpus(self, chunked):
        """A corpus-initial match has nothing to its left."""
        results = SearchResults(chunked, "", [Match(Span(0, 1), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [[]]
        assert conc["token_right_context"].to_list() == [["cat", "sat", "."]]

    def test_match_at_the_end_of_the_corpus(self, chunked):
        results = SearchResults(chunked, "", [Match(Span(7, 8), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_right_context"].to_list() == [[]]

    def test_null_tag_ends_the_context(self):
        """A null is a chunk boundary, like any other tag that isn't "I"."""
        df = pl.DataFrame(
            {
                "token": ["a", "b", "c", "d", "e"],
                "chunks": ["B", None, "I", "I", "I"],
            }
        )
        results = SearchResults(df, "", [Match(Span(3, 4), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["b", "c"]]

    @pytest.mark.parametrize(
        "dtype", [pl.String, pl.Categorical, pl.Enum(["B", "I"])], ids=str
    )
    def test_tag_dtypes(self, chunked, dtype):
        """A dictionary-encoded tag column holds the same tags."""
        df = chunked.with_columns(pl.col("chunks").cast(dtype))
        results = SearchResults(df, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["the", "cat"]]

    def test_non_tag_column_rejected(self, chunked):
        df = chunked.with_columns(pl.Series("chunks", range(8)))
        results = SearchResults(df, "", [Match(Span(2, 3), {})])

        with pytest.raises(ValueError, match="chunk_column must hold the chunk tags"):
            results.concordance("token", chunk_column="chunks")

    def test_window_is_ignored(self, chunked):
        results = SearchResults(chunked, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks", window=1)

        assert conc["token_left_context"].to_list() == [["the", "cat"]]


class TestArgumentChecks:
    """A misspelled column is reported here, not out of a query plan"""

    def test_missing_expr_column(self, results):
        with pytest.raises(ValueError, match="the corpus has no column 'toekn'"):
            results.concordance("toekn", window=2)

    def test_missing_chunk_column(self, results):
        with pytest.raises(ValueError, match="no column 'chunkz'"):
            results.concordance("token", chunk_column="chunkz")

    def test_missing_metadata_column(self, results):
        with pytest.raises(ValueError, match="no column 'flie_id'"):
            results.concordance("token", window=2, metadata="flie_id")

    def test_missing_column_names_the_parameter(self, results):
        with pytest.raises(ValueError, match="Use chunk_column="):
            results.concordance("token", chunk_column="chunkz")


class TestCollocates:
    """Collocate frequencies, laid out the way the association measures read them"""

    def test_frequencies(self, results):
        colloc = results.collocates("token", window=2, min_freq=1).unnest("freqs")
        by_token = {
            row["collocate"]: row
            for row in colloc.sort("collocate").iter_rows(named=True)
        }

        # "the" is in the window of both matches, twice for each.
        assert by_token["the"]["f12"] == 4
        # f1 is the window positions the matches ask for: 2 matches * 2 * 2.
        assert by_token["the"]["f1"] == 8
        # f2 is the corpus frequency of the collocate, n the corpus size.
        assert by_token["the"]["f2"] == 4
        assert by_token["the"]["n"] == 14

    def test_min_freq(self, results):
        kept = results.collocates("token", window=2, min_freq=4)

        assert kept["collocate"].to_list() == ["the"]

    def test_min_freq_zero_keeps_everything(self, results):
        all_of_them = results.collocates("token", window=2, min_freq=0)

        assert set(all_of_them["collocate"]) == {"the", "cat", "dog", "on"}

    @pytest.mark.parametrize("min_freq", [-1, None, 1.5])
    def test_bad_min_freq(self, results, min_freq):
        with pytest.raises(ValueError, match="min_freq must be a non-negative integer"):
            results.collocates("token", min_freq=min_freq)

    def test_window_zero(self, results):
        with pytest.raises(ValueError, match="window must be at least 1"):
            results.collocates("token", window=0)

    def test_several_columns_rejected(self, results):
        with pytest.raises(ValueError, match="must name a single column"):
            results.collocates(["token", "pos"])

    def test_expression(self, results):
        colloc = results.collocates(
            pl.col("token").str.to_uppercase(), window=2, min_freq=1
        )

        assert "THE" in colloc["collocate"].to_list()

    def test_null_is_not_a_collocate(self):
        """Neither a null token nor the null an empty context explodes to."""
        df = pl.DataFrame({"token": ["cat", "the", None]})
        results = SearchResults(df, "", [Match(Span(0, 1), {})])
        colloc = results.collocates("token", window=2, min_freq=0)

        assert colloc["collocate"].to_list() == ["the"]


class TestFunctionalInterface:
    """The free functions and the methods are the same call"""

    def test_concordance(self, results):
        from polars_corpus import concordance

        assert concordance(results, "token", window=2).equals(
            results.concordance("token", window=2)
        )

    def test_collocates(self, results):
        from polars_corpus import collocates

        free = collocates(results, "token", window=2, min_freq=1).sort("collocate")
        method = results.collocates("token", window=2, min_freq=1).sort("collocate")

        assert free.equals(method)


class TestSlicing:
    """head, tail, sample and shuffle agree on what they take and reject"""

    @pytest.fixture
    def ten(self):
        df = corpus(token=" ".join(f"t{i}" for i in range(10)))
        return SearchResults(df, "", [Match(Span(i, i + 1), {}) for i in range(10)])

    @pytest.mark.parametrize(
        "method,n,expected",
        [
            ("head", 3, [0, 1, 2]),
            ("head", 0, []),
            ("head", 99, list(range(10))),
            ("tail", 3, [7, 8, 9]),
            ("tail", 0, []),
            ("tail", 99, list(range(10))),
        ],
    )
    def test_slicing(self, ten, method, n, expected):
        sliced = getattr(ten, method)(n)

        assert [m.span.start for m in sliced._matches] == expected

    @pytest.mark.parametrize("method", ["head", "tail", "sample"])
    def test_negative_n(self, ten, method):
        with pytest.raises(ValueError, match="must be a non-negative integer"):
            getattr(ten, method)(-1)

    def test_sample_more_than_there_are(self, ten):
        with pytest.raises(ValueError, match="cannot sample 11 of 10 matches"):
            ten.sample(11)

    @pytest.mark.parametrize("method", ["sample", "shuffle"])
    def test_seed_leaves_the_global_state_alone(self, ten, method):
        import random

        random.seed(0)
        before = random.random()
        random.seed(0)
        getattr(ten, method)(*([5] if method == "sample" else []), seed=42)

        assert random.random() == before

    def test_sample_is_reproducible(self, ten):
        assert [m.span.start for m in ten.sample(5, seed=42)._matches] == [
            m.span.start for m in ten.sample(5, seed=42)._matches
        ]

    def test_shuffle_keeps_every_match(self, ten):
        shuffled = ten.shuffle(seed=42)

        assert sorted(m.span.start for m in shuffled._matches) == list(range(10))
