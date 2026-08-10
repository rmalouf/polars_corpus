"""Tests for embeddings: encoding text, encoding matches in context, centroids."""

import polars as pl
import pytest
from polars_corpus import Match, SearchResults, Span, centroid, encode

from .helpers import corpus

DIM = 3


class StubModel:
    """A SentenceTransformer stand-in.

    Records the strings it is handed, so a test can assert on the text that
    reached the model, and returns one vector per string, filled with that
    string's length so the rows stay distinguishable.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.seen: list[str] = []

    def get_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, normalize_embeddings=True):
        self.seen.extend(texts)
        return pl.Series(
            [[float(len(text))] * self.dim for text in texts],
            dtype=pl.Array(pl.Float32, self.dim),
        )


@pytest.fixture
def model():
    return StubModel()


def vectors(values: list[list[float]], dim: int = 2) -> pl.DataFrame:
    """A one-column frame of embedding vectors, as `encode` would leave them."""
    return pl.DataFrame({"vector": values}).cast({"vector": pl.Array(pl.Float32, dim)})


class TestEncode:
    """A text column encoded into fixed-width vectors"""

    def test_one_vector_per_row(self, model):
        df = corpus(token="the cat sat")
        out = df.select(encode(model, "token"))

        assert out.height == 3
        assert out["token"].dtype == pl.Array(pl.Float32, DIM)
        assert out["token"].to_list() == [[3.0] * DIM, [3.0] * DIM, [3.0] * DIM]

    def test_dimension_comes_from_the_model(self):
        df = corpus(token="the cat")
        out = df.select(encode(StubModel(dim=7), "token"))

        assert out["token"].dtype == pl.Array(pl.Float32, 7)

    def test_takes_an_expression(self, model):
        df = corpus(token="the cat")
        df.select(encode(model, pl.col("token").str.to_uppercase()))

        assert model.seen == ["THE", "CAT"]


class TestEncodeMatches:
    """Matches encoded together with the context they were used in"""

    @pytest.fixture
    def results(self):
        """The two "sat" matches, at positions 2 and 9."""
        df = corpus(token="the cat sat on the mat . the dog sat on the log .")
        return SearchResults(df, "sat", [Match(Span(2, 3), {}), Match(Span(9, 10), {})])

    def test_context_is_encoded_with_the_match(self, results, model):
        out = results.encode(model, window=2)

        assert out["token"].to_list() == [
            "the cat sat on the",
            "the dog sat on the",
        ]
        assert model.seen == ["the cat sat on the", "the dog sat on the"]

    def test_default_window_gives_context(self, results, model):
        """The default has to be a window the concordance actually builds."""
        out = results.encode(model)

        assert out["token"].to_list() == [
            "the cat sat on the mat . the",
            "the mat . the dog sat on the log .",
        ]

    def test_window_zero_is_the_bare_match(self, results, model):
        out = results.encode(model, window=0)

        assert out["token"].to_list() == ["sat", "sat"]

    def test_empty_context_is_not_padded(self, model):
        """A match at the edge of the corpus has no space to spare on that side."""
        df = corpus(token="the cat sat")
        results = SearchResults(df, "", [Match(Span(0, 1), {}), Match(Span(2, 3), {})])
        out = results.encode(model, window=5)

        assert out["token"].to_list() == ["the cat sat", "the cat sat"]

    def test_chunk_column(self, model):
        df = corpus(token="the cat sat . the dog sat .", chunks="B I I I B I I I")
        results = SearchResults(df, "", [Match(Span(6, 7), {})])
        out = results.encode(model, chunk_column="chunks")

        assert out["token"].to_list() == ["the dog sat ."]

    def test_vector_column(self, results, model):
        out = results.encode(model, window=2)

        assert out.columns == ["token", "vector"]
        assert out["vector"].dtype == pl.Array(pl.Float32, DIM)
        # The stub fills each vector with the length of the string it encoded.
        assert out["vector"].to_list() == [[18.0] * DIM, [18.0] * DIM]

    @pytest.mark.parametrize("metadata", ["file_id", ["file_id"]], ids=["str", "list"])
    def test_metadata_reaches_the_result(self, metadata):
        df = pl.DataFrame(
            {"token": ["a", "b", "c", "d"], "file_id": ["1", "1", "2", "2"]}
        )
        results = SearchResults(df, "", [Match(Span(1, 2), {}), Match(Span(3, 4), {})])
        out = results.encode(StubModel(), window=1, metadata=metadata)

        assert out.columns == ["token", "file_id", "vector"]
        assert out["file_id"].to_list() == ["1", "2"]

    def test_takes_an_expression(self, results, model):
        out = results.encode(model, pl.col("token").str.to_uppercase(), window=1)

        assert out.columns == ["token", "vector"]
        assert out["token"].to_list() == ["CAT SAT ON", "DOG SAT ON"]

    def test_list_expr_rejected(self, results, model):
        with pytest.raises(ValueError, match="expr must name a single column"):
            results.encode(model, ["token", "token"])

    def test_missing_column(self, results, model):
        with pytest.raises(ValueError, match="the corpus has no column 'toekn'"):
            results.encode(model, "toekn")


class TestCentroid:
    """Vectors averaged down to one, and scaled back to unit length"""

    def test_mean_of_the_vectors(self):
        df = vectors([[1.0, 2.0], [3.0, 6.0]])
        out = df.select(centroid("vector", normalize=False))

        assert out.columns == ["centroid"]
        assert out["centroid"].to_list() == [[2.0, 4.0]]

    def test_normalized_to_unit_length(self):
        df = vectors([[3.0, 4.0], [3.0, 4.0]])
        out = df.select(centroid("vector"))

        # The mean is (3, 4), of length 5.
        assert out["centroid"].to_list() == [[pytest.approx(0.6), pytest.approx(0.8)]]

    def test_aggregates_per_group(self):
        df = vectors([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]]).with_columns(
            group=pl.Series(["a", "a", "b"])
        )
        out = df.group_by("group", maintain_order=True).agg(
            centroid("vector", normalize=False)
        )

        assert out["centroid"].to_list() == [[2.0, 0.0], [0.0, 2.0]]

    def test_zero_centroid_is_left_alone(self):
        """Opposed vectors average to zero, which has no direction to scale."""
        df = vectors([[1.0, 0.0], [-1.0, 0.0]])
        out = df.select(centroid("vector"))

        assert out["centroid"].to_list() == [[0.0, 0.0]]

    def test_dtype_is_preserved(self):
        df = vectors([[1.0, 2.0], [3.0, 6.0]])
        out = df.select(centroid("vector"))

        assert out["centroid"].dtype == pl.Array(pl.Float32, 2)

    def test_round_trips_with_encode(self):
        """The centroid of encoded matches, which is what the pair is for."""
        df = corpus(token="the cat sat on the mat")
        results = SearchResults(df, "", [Match(Span(1, 2), {}), Match(Span(4, 5), {})])
        encoded = results.encode(StubModel(), window=1)
        out = encoded.select(centroid("vector", normalize=False))

        assert out["centroid"].dtype == pl.Array(pl.Float32, DIM)
        assert out.height == 1
