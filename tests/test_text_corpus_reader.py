import tempfile
from pathlib import Path

import polars as pl
import pytest

from nlpolars import read_text_corpus, scan_text_corpus
from nlpolars.io import TextCorpusReader


@pytest.fixture
def sample_file():
    content = """The/DT quick/JJ brown/NN fox/NN
jumps/VBZ over/IN the/DT lazy/JJ dog/NN

Another/DT sentence/NN here/RB
"""
    path = create_temp_file_with_content(content)
    yield path
    path.unlink()


@pytest.fixture
def malformed_file():
    # Missing a tag on one token
    content = """Good/JJ morning/NN everyone
MalformedLine noSlashHere
"""
    path = create_temp_file_with_content(content)
    yield path
    path.unlink()


@pytest.fixture
def multiple_files():
    files = []
    contents = ["First/DT file/NN\n", "Second/DT one/CD\nThird/JJ one/NN too/RB\n"]
    for content in contents:
        f = create_temp_file_with_content(content)
        files.append(f)

    yield files

    for f in files:
        f.unlink()


def create_temp_file_with_content(content: str) -> Path:
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write(content)
        f.flush()
        return Path(f.name)


def test_read_text_corpus(sample_file):
    df = read_text_corpus([sample_file])
    assert df.shape[1] == 3
    assert set(df.columns) == {"token", "tag", "sent"}
    assert df.filter(pl.col("sent") == "B").height == 3


def test_read_file_token_tag_parsing(sample_file):
    reader = TextCorpusReader([sample_file])
    rows = list(reader.read_file(sample_file))
    assert rows[0]["token"] == "The"
    assert rows[0]["tag"] == "DT"
    assert rows[0]["sent"] == "B"
    assert rows[1]["sent"] == "I"


def test_scan_text_corpus_collect(sample_file):
    lf = scan_text_corpus([sample_file])
    df = lf.collect()
    assert isinstance(df, pl.DataFrame)
    assert "token" in df.columns
    assert df.filter(pl.col("sent") == "B").height == 3


def test_scan_text_corpus_predicate_and_columns(sample_file):
    lf = scan_text_corpus([sample_file])
    filtered = lf.select(["token"]).filter(pl.col("token").str.contains("fox|dog"))
    df = filtered.collect()
    assert df.shape[1] == 1
    assert all(t in {"fox", "dog"} for t in df["token"])


def test_handles_malformed_lines_gracefully(malformed_file):
    reader = TextCorpusReader([malformed_file])
    with pytest.raises(ValueError):
        list(reader.read_file(malformed_file))


def test_multiple_files_combined(multiple_files):
    df = read_text_corpus(multiple_files)
    assert df.filter(pl.col("token") == "First").height == 1
    assert df.filter(pl.col("token") == "Third").height == 1


def test_empty_file():
    path = create_temp_file_with_content("")
    df = read_text_corpus([path])
    assert df.is_empty()
    path.unlink()
