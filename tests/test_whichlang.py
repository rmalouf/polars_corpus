import polars as pl
import pytest
from polars.polars import ComputeError

from polars_text import whichlang


@pytest.fixture
def sample_df():
    yield pl.DataFrame(
        {
            "index": [1, 2, 3, 4, 5],
            "text": ["This is a test.", "이건 테스트야", "Dies ist ein Test", "", " "],
            "lang": ["eng", "kor", "deu", "eng", "ita"],
        }
    )


def test_whichlang_dataframe(sample_df):
    df = sample_df.with_columns(pred=pl.col("text").text.whichlang())
    assert len(df) == 5
    assert all(df["lang"] == df["pred"])

    df = sample_df.with_columns(pred=whichlang("text"))
    assert len(df) == 5
    assert all(df["lang"] == df["pred"])


def test_whichlang_lazyframe(sample_df):
    df = sample_df.lazy().with_columns(pred=pl.col("text").text.whichlang())
    assert isinstance(df, pl.LazyFrame)
    df = df.collect()
    assert len(df) == 5
    assert all(df["lang"] == df["pred"])

    df = sample_df.lazy().with_columns(pred=whichlang("text"))
    assert isinstance(df, pl.LazyFrame)
    df = df.collect()
    assert len(df) == 5
    assert all(df["lang"] == df["pred"])


def test_whichlang_dataframe_error(sample_df):
    with pytest.raises(ComputeError):
        df = sample_df.with_columns(pred=pl.col("index").text.whichlang())
    with pytest.raises(ComputeError):
        df = sample_df.with_columns(pred=whichlang("index"))


def test_whichlang_lazyframe_error(sample_df):
    df = sample_df.lazy().with_columns(pred=pl.col("index").text.whichlang())
    assert isinstance(df, pl.LazyFrame)
    with pytest.raises(ComputeError):
        df = df.collect()

    df = sample_df.lazy().with_columns(pred=whichlang("index"))
    assert isinstance(df, pl.LazyFrame)
    with pytest.raises(ComputeError):
        df = df.collect()
