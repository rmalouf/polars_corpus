"""Out-of-core search: a lazy corpus, chunked by file, must agree with eager."""

import polars as pl
import polars_corpus as plc
import pytest
from polars_corpus import LazySearchResults, scan_text_corpus
from polars_corpus.matcher import search, search_cqp

from .helpers import corpus

# Four files -- contiguous runs, deliberately not sorted by id -- with "brown
# fox" both inside a file and straddling the d3/d1 boundary.
FILES = corpus(
    token="the quick brown fox brown fox . and a fox the lazy fox and the fox",
    file_id="d3 d3 d3 d3 d1 d1 d1 d1 d2 d2 d4 d4 d4 d4 d4 d4",
)

QUERIES = [
    pytest.param('[token="fox"]', id="single-token"),
    pytest.param('[token="brown"] [token="fox"]', id="sequence"),
    pytest.param('[token="the"] []* [token="fox"]', id="quantified-gap"),
    pytest.param('$t: ([token="the"]?) [token="fox"]', id="optional-binding"),
    pytest.param('[token="wolf"]', id="no-matches"),
]

# One file per chunk, a few files per chunk, everything in one chunk.
CHUNK_SIZES = [1, 6, 10**9]


def eager_and_lazy(query, chunk_tokens, df=FILES):
    return (
        search_cqp(df, query, file_id_column="file_id"),
        search_cqp(
            df.lazy(), query, file_id_column="file_id", chunk_tokens=chunk_tokens
        ),
    )


@pytest.mark.parametrize("chunk_tokens", CHUNK_SIZES)
@pytest.mark.parametrize("query", QUERIES)
def test_concordance_matches_eager(query, chunk_tokens):
    eager, lazy = eager_and_lazy(query, chunk_tokens)

    if eager is None:
        assert lazy is None
        return
    assert isinstance(lazy, LazySearchResults)
    assert len(lazy) == len(eager)
    assert lazy.variables == eager.variables
    assert lazy.concordance("token", window=3, metadata="file_id").equals(
        eager.concordance("token", window=3, metadata="file_id")
    )


@pytest.mark.parametrize("chunk_tokens", CHUNK_SIZES)
def test_as_str_concordance_matches_eager(chunk_tokens):
    """Joining happens after the chunks are stitched back together."""
    eager, lazy = eager_and_lazy('[token="fox"]', chunk_tokens)
    conc = lazy.concordance("token", window=3, as_str=True)

    assert conc["token"].dtype == pl.String
    assert conc.equals(eager.concordance("token", window=3, as_str=True))


@pytest.mark.parametrize("chunk_tokens", CHUNK_SIZES)
def test_collocates_match_eager(chunk_tokens):
    eager, lazy = eager_and_lazy('[token="fox"]', chunk_tokens)

    assert (
        lazy.collocates("token", window=2, min_freq=0)
        .sort("collocate")
        .equals(eager.collocates("token", window=2, min_freq=0).sort("collocate"))
    )


@pytest.mark.parametrize("chunk_tokens", CHUNK_SIZES)
def test_spans_as_chunks_match_eager(chunk_tokens):
    eager, lazy = eager_and_lazy('[token="brown"] [token="fox"]', chunk_tokens)
    tagged = lazy.with_spans_as_chunks()

    assert isinstance(tagged, pl.LazyFrame)
    assert tagged.collect().equals(eager.with_spans_as_chunks())


def test_spans_as_chunks_keeps_corpus_order():
    """The tags have to land on the tokens they were read off.

    Tagging joins the matches back onto the corpus by position, and a join is
    free to return its rows in any order. A small frame comes back in order
    whatever it does, so this needs a corpus big enough to be split up: at
    200_000 tokens an unordered join already scrambles it.
    """
    n = 500_000
    tokens = ["fox" if i % 1000 == 500 else "the" for i in range(n)]
    corpus = pl.DataFrame(
        {"token": tokens, "file_id": [f"d{i // 5000}" for i in range(n)]}
    )
    tagged = plc.search(corpus.lazy(), "fox").with_spans_as_chunks().collect()

    assert tagged["token"].to_list() == tokens
    assert (tagged["spans"] != "O").sum() == n // 1000


@pytest.mark.parametrize(
    "dtype", [pl.String, pl.Categorical, pl.Enum(["d1", "d2", "d3", "d4"]), pl.UInt32]
)
def test_file_id_dtypes(dtype):
    """Chunking and rejoining survive however file ids are stored."""
    cast = pl.col("file_id")
    if dtype == pl.UInt32:
        cast = cast.str.strip_prefix("d")
    df = FILES.with_columns(cast.cast(dtype))
    eager, lazy = eager_and_lazy('[token="brown"] [token="fox"]', 6, df)

    assert lazy.concordance("token", window=2, metadata="file_id").equals(
        eager.concordance("token", window=2, metadata="file_id")
    )


def test_chunk_column_concordance_matches_eager():
    df = FILES.with_columns(sent=pl.Series("B I I I B I I I B I B I I I I I".split()))
    eager, lazy = eager_and_lazy('[token="fox"]', 6, df)

    assert lazy.concordance("token", chunk_column="sent").equals(
        eager.concordance("token", chunk_column="sent")
    )


def test_match_frame_spans_are_file_relative():
    _, lazy = eager_and_lazy('[token="brown"] [token="fox"]', 1)

    matches = lazy._matches
    # The id itself is not carried: `_file` indexes the files frame, and that
    # is what says where the offsets count from.
    assert lazy._files["file_id"].gather(matches["_file"]).to_list() == ["d3", "d1"]
    # d3's match sits at its file's start + 2; d1's at its own start.
    assert matches["start"].to_list() == [2, 0]
    assert matches["end"].to_list() == [4, 2]


class TestSlicing:
    @pytest.fixture
    def lazy(self):
        return eager_and_lazy('[token="fox"]', 6)[1]

    def test_head_tail(self, lazy):
        assert len(lazy.head(2)) == 2
        assert len(lazy.tail(2)) == 2
        assert lazy.head(99) is lazy
        with pytest.raises(ValueError, match="non-negative"):
            lazy.head(-1)

    def test_sample(self, lazy):
        assert len(lazy.sample(3, seed=42)) == 3
        with pytest.raises(ValueError, match="cannot sample"):
            lazy.sample(99)

    def test_shuffle_keeps_every_match(self, lazy):
        shuffled = lazy.shuffle(seed=42)

        assert sorted(shuffled._matches["start"].to_list()) == sorted(
            lazy._matches["start"].to_list()
        )

    def test_sliced_concordance_keeps_the_slice_order(self, lazy):
        shuffled = lazy.shuffle(seed=42)
        conc = shuffled.concordance("token", metadata="file_id")

        expected = shuffled._files["file_id"].gather(shuffled._matches["_file"])
        assert conc["file_id"].to_list() == expected.to_list()

    @pytest.mark.parametrize(
        "method,args", [("sample", (3,)), ("shuffle", ())], ids=["sample", "shuffle"]
    )
    def test_seed_agrees_with_eager(self, method, args):
        """Both paths draw from Python's random, so a seed picks the same matches."""
        eager, lazy = eager_and_lazy('[token="fox"]', 6)

        assert (
            getattr(lazy, method)(*args, seed=7)
            .concordance("token", metadata="file_id")
            .equals(
                getattr(eager, method)(*args, seed=7).concordance(
                    "token", metadata="file_id"
                )
            )
        )

    def test_seed_leaves_the_global_state_alone(self, lazy):
        import random

        random.seed(0)
        before = random.random()
        random.seed(0)
        lazy.sample(3, seed=42)

        assert random.random() == before


def test_simple_query_language():
    df = corpus(
        token="the quick fox the slow dog",
        pos="DT JJ NN DT JJ NN",
        file_id="f1 f1 f1 f2 f2 f2",
    )
    results = search(df.lazy(), "the _JJ", file_id_column="file_id")

    assert len(results) == 2
    assert results._query == "the _JJ"


def test_lazyframe_namespace():
    results = FILES.lazy().corpus.search_cqp('[token="fox"]', file_id_column="file_id")

    assert isinstance(results, LazySearchResults)


@pytest.mark.parametrize(
    "file_ids,chunk_tokens",
    [
        pytest.param(["f1", "f1", "f2", "f1"], 10**9, id="within-one-chunk"),
        # The plan says f1:2 then f2:2, so each chunk's boundary falls mid-run.
        pytest.param(["f1", "f2", "f1", "f2"], 2, id="alternating"),
        # A repeated run sitting exactly on a planned chunk boundary.
        pytest.param(["f1", "f1", "f2", "f2", "f1", "f1"], 2, id="run-at-boundary"),
    ],
)
def test_interleaved_file_ids_rejected(file_ids, chunk_tokens):
    df = pl.DataFrame({"token": ["a"] * len(file_ids), "file_id": file_ids})

    with pytest.raises(ValueError, match="interleaves"):
        search_cqp(
            df.lazy(),
            '[token="a"]',
            file_id_column="file_id",
            chunk_tokens=chunk_tokens,
        )


def test_null_file_id_forms_its_own_file():
    """A contiguous run of nulls is a file like any other, as in eager search."""
    df = pl.DataFrame(
        {"token": ["a", "a", "a", "a"], "file_id": ["f1", "f1", None, None]}
    )
    results = search_cqp(df.lazy(), '[token="a"] [token="a"]', file_id_column="file_id")

    # One match per file; none across the f1/null boundary.
    assert len(results) == 2


@pytest.mark.parametrize(
    "df,file_id_column",
    [
        pytest.param(FILES, None, id="ids-ignored"),
        pytest.param(FILES.drop("file_id"), "file_id", id="no-ids-to-find"),
    ],
)
def test_lazy_without_file_ids_is_searched_as_one_chunk(df, file_id_column):
    """Nothing to chunk on, so the corpus is one chunk -- but still out of core."""
    query = '[token="brown"] [token="fox"]'
    lazy = search_cqp(df.lazy(), query, file_id_column)

    assert isinstance(lazy, LazySearchResults)
    assert lazy.concordance("token", window=3).equals(
        search_cqp(df, query, file_id_column).concordance("token", window=3)
    )


@pytest.mark.parametrize(
    "df,file_id_column",
    [
        pytest.param(FILES, None, id="ids-ignored"),
        pytest.param(FILES.drop("file_id"), "file_id", id="no-ids-to-find"),
    ],
)
def test_lazy_without_file_ids_tags_spans(df, file_id_column):
    """One file spanning the corpus, so a match's offsets are its positions."""
    query = '[token="brown"] [token="fox"]'
    lazy = search_cqp(df.lazy(), query, file_id_column)

    assert (
        lazy.with_spans_as_chunks()
        .collect()
        .equals(search_cqp(df, query, file_id_column).with_spans_as_chunks())
    )


def test_lazy_without_file_ids_collocates():
    """The counts have no file ids to report a range over, and say so."""
    lazy = search_cqp(FILES.drop("file_id").lazy(), '[token="fox"]')
    eager = search_cqp(FILES.drop("file_id"), '[token="fox"]')

    result = plc.collocations(lazy, "token", "freq", window=2, min_freq=1)
    assert result.columns == ["collocate", "freqs", "freq"]
    assert result.sort("collocate").equals(
        plc.collocations(eager, "token", "freq", window=2, min_freq=1).sort("collocate")
    )


def test_lazy_without_file_ids_ignores_chunk_tokens():
    """There is no safe place to cut, so the corpus stays one chunk."""
    query = '[token="fox"] [token="brown"]'
    # A cut every two tokens would fall straight through this match.
    results = search_cqp(FILES.lazy(), query, file_id_column=None, chunk_tokens=2)

    assert len(results) == 1


def test_lazy_without_file_ids_matches_across_them():
    """One continuous run of tokens: a match may straddle a file boundary."""
    query = '[token="fox"] [token="brown"]'

    assert search_cqp(FILES.lazy(), query, file_id_column="file_id") is None
    assert len(search_cqp(FILES.lazy(), query, file_id_column=None)) == 1


def test_named_file_id_column_must_exist():
    """Only the default name is optional; one the caller asked for is checked.

    The name is wrong whatever the corpus holds, so the error comes before it
    is read: the UDF here would raise if the search reached the tokens.
    """

    def unread(_: pl.DataFrame) -> pl.DataFrame:
        raise AssertionError("the corpus was read")

    lf = FILES.lazy().map_batches(unread, schema=FILES.schema)

    with pytest.raises(ValueError, match="no column 'doc'"):
        search_cqp(lf, '[token="fox"]', file_id_column="doc")


def test_bad_chunk_tokens():
    with pytest.raises(ValueError, match="chunk_tokens"):
        search_cqp(FILES.lazy(), '[token="fox"]', "file_id", chunk_tokens=0)


@pytest.mark.parametrize("file_id_column", ["file_id", None])
def test_empty_lazy_corpus(file_id_column):
    lf = pl.DataFrame({"token": [], "file_id": []}).lazy()

    assert search_cqp(lf, '[token="fox"]', file_id_column) is None


def test_scan_text_corpus_is_searchable(tmp_path):
    """End to end: scanned text files are searched without read_text_corpus."""
    for i, text in enumerate(["The/DT quick/JJ fox/NN\n", "A/DT lazy/JJ dog/NN\n"]):
        (tmp_path / f"corpus{i}.txt").write_text(text)
    lf = scan_text_corpus(sorted(tmp_path.glob("*.txt")))
    results = plc.search(lf, "_JJ _NN", file_id_column="file_id")

    assert len(results) == 2
    conc = results.concordance("token", metadata="file_id")
    assert conc["token"].to_list() == [["quick", "fox"], ["lazy", "dog"]]
