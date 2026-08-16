"""Tests for concordancing: context windows, collocates, and result slicing."""

import polars as pl
import polars_corpus as plc
import pytest
from polars_corpus import Match, Span

from .helpers import corpus, search_results

# "the cat sat on the mat . the dog sat on the log ."
TOKENS = "the cat sat on the mat . the dog sat on the log ."


@pytest.fixture
def results():
    """The two "sat" matches, at positions 2 and 9."""
    df = corpus(token=TOKENS)
    return search_results(df, "sat", [Match(Span(2, 3), {}), Match(Span(9, 10), {})])


def test_lazy_corpus_rejected():
    """Every method here reads the corpus, so it is checked at construction."""
    df = corpus(token=TOKENS)
    with pytest.raises(ValueError, match="must be an eager polars DataFrame"):
        search_results(df.lazy(), "sat", [Match(Span(2, 3), {})])


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
        results = search_results(df, "", [Match(Span(0, 1), {}), Match(Span(2, 3), {})])
        conc = results.concordance("token", window=5)

        # Neither context is padded out to the five tokens asked for.
        assert conc["token_left_context"].to_list() == [[], ["the", "cat"]]
        assert conc["token_right_context"].to_list() == [["cat", "sat"], []]

    def test_context_runs_across_files_without_file_ids(self):
        """Results that know no file_id_column have no boundaries to stop at."""
        df = pl.DataFrame(
            {"token": ["a", "b", "c", "d"], "file_id": ["1", "1", "2", "2"]}
        )
        results = search_results(df, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", window=2)

        assert conc["token_left_context"].to_list() == [["a", "b"]]

    def test_context_stops_at_file_boundaries(self):
        """With a file_id_column, context is clipped to the match's file."""
        df = pl.DataFrame(
            {"token": ["a", "b", "c", "d", "e"], "file_id": ["1", "1", "2", "2", "1"]}
        )
        results = search_results(
            df, "", [Match(Span(2, 3), {})], file_id_column="file_id"
        )
        conc = results.concordance("token", window=2)

        # Neither into the file before nor into the (resumed id) file after.
        assert conc["token_left_context"].to_list() == [[]]
        assert conc["token_right_context"].to_list() == [["d"]]

    def test_chunk_context_stops_at_file_boundaries(self):
        """A chunk running on across a file boundary is clipped there too."""
        df = pl.DataFrame(
            {
                "token": ["a", "b", "c", "d"],
                "file_id": ["1", "1", "2", "2"],
                "chunks": ["B", "I", "I", "I"],
            }
        )
        results = search_results(
            df, "", [Match(Span(2, 3), {})], file_id_column="file_id"
        )
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [[]]
        assert conc["token_right_context"].to_list() == [["d"]]

    @pytest.mark.parametrize("window", [-1, 1.5, None, "2", True])
    def test_bad_window(self, results, window):
        with pytest.raises(ValueError, match="window must be a non-negative integer"):
            results.concordance("token", window=window)

    def test_several_columns(self):
        df = corpus(token="the cat sat", pos="DT NN VB")
        results = search_results(df, "", [Match(Span(1, 2), {})])
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
        results = search_results(chunked, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["the", "cat"]]
        assert conc["token_right_context"].to_list() == [["."]]

    def test_context_stops_at_the_next_chunk(self, chunked):
        results = search_results(chunked, "", [Match(Span(6, 7), {})])
        conc = results.concordance("token", chunk_column="chunks")

        # The second sentence only, not back over the first.
        assert conc["token_left_context"].to_list() == [["the", "dog"]]

    def test_match_at_the_start_of_the_corpus(self, chunked):
        """A corpus-initial match has nothing to its left."""
        results = search_results(chunked, "", [Match(Span(0, 1), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [[]]
        assert conc["token_right_context"].to_list() == [["cat", "sat", "."]]

    def test_match_at_the_end_of_the_corpus(self, chunked):
        results = search_results(chunked, "", [Match(Span(7, 8), {})])
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
        results = search_results(df, "", [Match(Span(3, 4), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["b", "c"]]

    @pytest.mark.parametrize(
        "dtype", [pl.String, pl.Categorical, pl.Enum(["B", "I"])], ids=str
    )
    def test_tag_dtypes(self, chunked, dtype):
        """A dictionary-encoded tag column holds the same tags."""
        df = chunked.with_columns(pl.col("chunks").cast(dtype))
        results = search_results(df, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks")

        assert conc["token_left_context"].to_list() == [["the", "cat"]]

    def test_non_tag_column_rejected(self, chunked):
        df = chunked.with_columns(pl.Series("chunks", range(8)))
        results = search_results(df, "", [Match(Span(2, 3), {})])

        with pytest.raises(ValueError, match="chunk_column must hold the chunk tags"):
            results.concordance("token", chunk_column="chunks")

    def test_window_is_ignored(self, chunked):
        results = search_results(chunked, "", [Match(Span(2, 3), {})])
        conc = results.concordance("token", chunk_column="chunks", window=1)

        assert conc["token_left_context"].to_list() == [["the", "cat"]]


class TestBindings:
    """The `$name:` bindings a query captured, as columns of their own"""

    @pytest.fixture
    def bound(self):
        """ "the big cat" and "the old dog", with the adjective and noun bound."""
        df = corpus(
            token="the big cat sat . the old dog sat .",
            pos="DT JJ NN VB . DT JJ NN VB .",
        )
        return plc.search_cqp(
            df, '[pos="DT"] $adj: [pos="JJ"] $noun: [pos="NN"]', file_id_column=None
        )

    def test_a_column_per_variable(self, bound):
        conc = bound.concordance("token", window=1)

        # The bound columns follow the match and its context, so the matched
        # column is still the first list column the widget finds.
        assert conc.columns == [
            "token_left_context",
            "token",
            "token_right_context",
            "token_adj",
            "token_noun",
        ]
        assert conc["token_adj"].to_list() == [["big"], ["old"]]
        assert conc["token_noun"].to_list() == [["cat"], ["dog"]]

    def test_variables_in_the_order_the_query_binds_them(self, bound):
        assert bound.variables == ["adj", "noun"]

    def test_every_expr_column_is_bound(self, bound):
        conc = bound.concordance(["token", "pos"])

        assert conc.columns == [
            "token",
            "token_adj",
            "token_noun",
            "pos",
            "pos_adj",
            "pos_noun",
        ]
        assert conc["pos_adj"].to_list() == [["JJ"], ["JJ"]]

    def test_bindings_reach_the_chunk_concordance(self, bound):
        df = bound._df.with_columns(chunks=pl.lit("I"))
        chunked = search_results(df, "", bound._matches, bound.variables)
        conc = chunked.concordance("token", chunk_column="chunks")

        assert conc["token_adj"].to_list() == [["big"], ["old"]]

    @pytest.mark.parametrize(
        "bindings,expected",
        [
            (True, ["token", "token_adj", "token_noun"]),
            (False, ["token"]),
            ("noun", ["token", "token_noun"]),
            (["noun", "adj"], ["token", "token_noun", "token_adj"]),
            (["noun", "noun"], ["token", "token_noun"]),
        ],
        ids=["all", "none", "one", "named-order", "repeat-dropped"],
    )
    def test_choosing_variables(self, bound, bindings, expected):
        assert bound.concordance("token", bindings=bindings).columns == expected

    def test_a_query_that_binds_nothing_adds_nothing(self, results):
        assert results.variables == []
        assert results.concordance("token", window=1).columns == [
            "token_left_context",
            "token",
            "token_right_context",
        ]

    def test_unbound_variable_is_null(self):
        """A branch of the query that never bound the name leaves a null."""
        df = corpus(token="the dog and the big cat", pos="DT NN CC DT JJ NN")
        results = plc.search_cqp(df, '[pos="DT"] ($mods: [pos="JJ"]+)? [pos="NN"]')
        conc = results.concordance("token")

        assert conc["token_mods"].to_list() == [None, ["big"]]

    def test_binding_that_matched_no_token_is_empty(self):
        """A zero-width binding is an empty list, as against an unbound null."""
        df = corpus(token="a b c")
        results = search_results(
            df, "", [Match(Span(0, 2), {"x": Span(0, 0)}), Match(Span(2, 3), {})]
        )

        assert results.concordance("token")["token_x"].to_list() == [[], None]

    def test_variables_of_hand_built_matches(self):
        """With no query to read the order off, the names are alphabetical."""
        df = corpus(token="a b c")
        results = search_results(
            df,
            "",
            [
                Match(Span(0, 1), {"z": Span(0, 1)}),
                Match(Span(1, 2), {"y": Span(1, 2)}),
            ],
        )

        assert results.variables == ["y", "z"]

    @pytest.mark.parametrize(
        "method,args", [("head", (1,)), ("sample", (1,)), ("shuffle", ())]
    )
    def test_slicing_keeps_the_variables(self, bound, method, args):
        assert getattr(bound, method)(*args).variables == ["adj", "noun"]

    def test_unknown_variable(self, bound):
        with pytest.raises(ValueError, match="no variable 'nuon'.*Did you mean 'noun'"):
            bound.concordance("token", bindings="nuon")

    def test_no_variables_to_name(self, results):
        with pytest.raises(ValueError, match="bound no variables"):
            results.concordance("token", bindings="adj")

    def test_bad_bindings_type(self, bound):
        with pytest.raises(ValueError, match="bindings must be True, False"):
            bound.concordance("token", bindings=3)


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
        results = search_results(df, "", [Match(Span(0, 1), {})])
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
        return search_results(df, "", [Match(Span(i, i + 1), {}) for i in range(10)])

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

    def test_len_counts_the_matches(self, ten):
        assert len(ten) == 10
        assert len(ten.head(3)) == 3

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
