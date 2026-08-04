import polars as pl
import pytest
from polars_corpus import Match, SearchResults, Span, with_chunk_index


def chunk_ids_via_function(df: pl.DataFrame) -> list:
    return with_chunk_index(df, "bio")["chunk_idx"].to_list()


def chunk_ids_via_expression(df: pl.DataFrame) -> list:
    result = df.with_columns(pl.col("bio").corpus.chunk_id().alias("chunk_idx"))
    return result["chunk_idx"].to_list()


@pytest.mark.parametrize(
    "how", [chunk_ids_via_function, chunk_ids_via_expression], ids=["function", "expr"]
)
@pytest.mark.parametrize(
    "bio,expected",
    [
        pytest.param(["B", "I", "O", "B", "I"], [1, 1, None, 2, 2], id="two-chunks"),
        pytest.param(["O", "O", "O"], [None, None, None], id="all-outside"),
        pytest.param(["B", "O", "B", "O"], [1, None, 2, None], id="single-token"),
        # An I with no preceding B falls into chunk 0 rather than erroring.
        pytest.param(["I", "I", "O"], [0, 0, None], id="malformed-leading-I"),
    ],
)
def test_chunk_index(how, bio, expected):
    df = pl.DataFrame({"bio": bio})
    assert how(df) == expected


@pytest.mark.parametrize(
    "how", [chunk_ids_via_function, chunk_ids_via_expression], ids=["function", "expr"]
)
def test_chunk_index_empty(how):
    df = pl.DataFrame({"bio": []}, schema={"bio": pl.Utf8})
    assert how(df) == []


def test_chunk_index_custom_name():
    df = pl.DataFrame({"bio": ["B", "I"]})
    result = with_chunk_index(df, "bio", name="my_chunk_idx")
    assert result["my_chunk_idx"].to_list() == [1, 1]


class TestChunkIdExpression:
    """Contexts the expression form has to work in, beyond with_columns"""

    def test_in_select(self):
        df = pl.DataFrame({"bio": ["B", "I", "O"]})
        result = df.select(pl.col("bio").corpus.chunk_id().alias("chunk_idx"))
        assert result["chunk_idx"].to_list() == [1, 1, None]

    def test_in_filter(self):
        df = pl.DataFrame(
            {"token": ["The", "quick", "brown", "fox"], "bio": ["B", "I", "O", "B"]}
        )
        result = df.filter(pl.col("bio").corpus.chunk_id().is_not_null())
        assert result["token"].to_list() == ["The", "quick", "fox"]

    def test_multiple_columns(self):
        df = pl.DataFrame({"bio1": ["B", "I", "O"], "bio2": ["O", "B", "I"]})
        result = df.with_columns(
            pl.col("bio1").corpus.chunk_id().alias("chunk1"),
            pl.col("bio2").corpus.chunk_id().alias("chunk2"),
        )
        assert result["chunk1"].to_list() == [1, 1, None]
        assert result["chunk2"].to_list() == [None, 1, 1]

    def test_with_lazyframe(self):
        df = pl.DataFrame({"bio": ["B", "I", "O"]})
        result = (
            df.lazy()
            .with_columns(pl.col("bio").corpus.chunk_id().alias("chunk_idx"))
            .collect()
        )
        assert result["chunk_idx"].to_list() == [1, 1, None]


class TestNgramsExpression:
    """Tests for the ngrams() expression method."""

    def test_bigrams(self):
        df = pl.DataFrame({"token": ["the", "quick", "brown", "fox"]})
        result = df.with_columns(pl.col("token").corpus.ngrams(2).alias("bigrams"))

        assert result.schema["bigrams"] == pl.Struct(
            [pl.Field("_0", pl.Utf8), pl.Field("_1", pl.Utf8)]
        )
        # The tail is padded with nulls rather than truncated.
        assert result["bigrams"].to_list() == [
            {"_0": "the", "_1": "quick"},
            {"_0": "quick", "_1": "brown"},
            {"_0": "brown", "_1": "fox"},
            {"_0": "fox", "_1": None},
        ]

    def test_trigrams(self):
        df = pl.DataFrame({"token": ["the", "quick", "brown", "fox", "jumps"]})
        result = df.with_columns(pl.col("token").corpus.ngrams(3).alias("trigrams"))

        assert result["trigrams"].to_list() == [
            {"_0": "the", "_1": "quick", "_2": "brown"},
            {"_0": "quick", "_1": "brown", "_2": "fox"},
            {"_0": "brown", "_1": "fox", "_2": "jumps"},
            {"_0": "fox", "_1": "jumps", "_2": None},
            {"_0": "jumps", "_1": None, "_2": None},
        ]

    def test_with_lazyframe(self):
        df = pl.DataFrame({"token": ["the", "quick", "brown"]})
        result = (
            df.lazy()
            .with_columns(pl.col("token").corpus.ngrams(2).alias("bigrams"))
            .collect()
        )
        assert result["bigrams"][0] == {"_0": "the", "_1": "quick"}

    def test_empty_dataframe(self):
        df = pl.DataFrame({"token": []}, schema={"token": pl.Utf8})
        result = df.with_columns(pl.col("token").corpus.ngrams(2).alias("bigrams"))

        assert len(result) == 0
        assert "bigrams" in result.columns


class TestWithSpansAsChunks:
    """Matched spans render back onto the corpus as BIO tags"""

    @pytest.mark.parametrize(
        "n_tokens,match_spans,expected",
        [
            pytest.param(5, [(1, 3)], ["O", "B", "I", "O", "O"], id="single-span"),
            pytest.param(
                6, [(1, 3), (4, 6)], ["O", "B", "I", "O", "B", "I"], id="two-spans"
            ),
            pytest.param(
                4, [(0, 1), (3, 4)], ["B", "O", "O", "B"], id="single-token-spans"
            ),
            pytest.param(
                4, [(0, 2), (2, 4)], ["B", "I", "B", "I"], id="adjacent-spans"
            ),
            pytest.param(3, [(0, 3)], ["B", "I", "I"], id="covers-whole-frame"),
            pytest.param(3, [], ["O", "O", "O"], id="no-matches"),
            pytest.param(0, [], [], id="empty-frame"),
        ],
    )
    def test_bio_tagging(self, n_tokens, match_spans, expected):
        df = pl.DataFrame({"token": [f"t{i}" for i in range(n_tokens)]})
        matches = [Match(Span(s, e), {}) for s, e in match_spans]
        result = SearchResults(df, "", matches).with_spans_as_chunks()

        assert result["spans"].to_list() == expected

    def test_custom_column_name(self):
        df = pl.DataFrame({"token": ["The", "quick"]})
        results = SearchResults(df, "", [Match(Span(0, 2), {})])
        result = results.with_spans_as_chunks(name="my_spans")

        assert result["my_spans"].to_list() == ["B", "I"]

    def test_out_of_bounds_span(self):
        df = pl.DataFrame({"token": ["The", "quick"]})
        results = SearchResults(df, "", [Match(Span(0, 5), {})])

        with pytest.raises(ValueError):
            results.with_spans_as_chunks()

    def test_negative_span_position(self):
        df = pl.DataFrame({"token": ["The", "quick"]})
        with pytest.raises(OverflowError):
            SearchResults(df, "", [Match(Span(-1, 1), {})])


@pytest.fixture
def metadata_corpus():
    return pl.DataFrame(
        {
            "token": ["The", "quick", "brown", "fox", "jumps"],
            "file_id": ["doc1", "doc1", "doc1", "doc2", "doc2"],
            "category": ["news", "news", "news", "fiction", "fiction"],
        }
    )


class TestConcordanceMetadata:
    """Tests for the metadata parameter of SearchResults.concordance."""

    def test_list_of_columns(self, metadata_corpus):
        results = SearchResults(
            metadata_corpus, "", [Match(Span(1, 2), {}), Match(Span(4, 5), {})]
        )  # "quick" and "jumps"
        conc = results.concordance("token", window=1, metadata=["file_id", "category"])

        assert conc["file_id"].to_list() == ["doc1", "doc2"]
        assert conc["category"].to_list() == ["news", "fiction"]

    def test_single_column_as_string(self, metadata_corpus):
        results = SearchResults(metadata_corpus, "", [Match(Span(1, 2), {})])
        conc = results.concordance("token", window=1, metadata="file_id")

        assert conc["file_id"].to_list() == ["doc1"]
        assert "category" not in conc.columns

    def test_no_metadata_by_default(self, metadata_corpus):
        results = SearchResults(metadata_corpus, "", [Match(Span(1, 2), {})])
        conc = results.concordance("token", window=1)

        assert "file_id" not in conc.columns

    def test_with_chunk_column(self, metadata_corpus):
        df = metadata_corpus.with_columns(
            pl.Series("chunks", ["O", "B", "I", "B", "I"])
        )
        results = SearchResults(df, "", [Match(Span(3, 5), {})])  # "fox jumps"
        conc = results.concordance("token", chunk_column="chunks", metadata="file_id")

        assert conc["file_id"].to_list() == ["doc2"]


@pytest.mark.parametrize(
    "dtype",
    [pl.Categorical, pl.Enum(["DT", "JJ", "NN", "VBZ", "doc1", "doc2"])],
)
def test_concordance_over_dictionary_column(dtype):
    """Categorical and Enum columns must survive the trip through Rust.

    Requires the dtype-categorical feature in Cargo.toml; without it the
    extension module panics converting dictionary arrays.
    """
    df = pl.DataFrame(
        {
            "token": ["The", "quick", "brown", "fox", "jumps"],
            "pos": ["DT", "JJ", "JJ", "NN", "VBZ"],
            "file_id": ["doc1", "doc1", "doc1", "doc2", "doc2"],
        }
    ).with_columns(pl.col("pos").cast(dtype), pl.col("file_id").cast(dtype))

    results = SearchResults(df, "", [Match(Span(1, 3), {})])
    conc = results.concordance("pos", window=1, metadata="file_id")

    assert conc["pos"].to_list() == [["JJ", "JJ"]]
    assert conc["pos_left_context"].to_list() == [["DT"]]
    assert conc["file_id"].to_list() == ["doc1"]
