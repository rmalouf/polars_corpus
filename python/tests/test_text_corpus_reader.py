from pathlib import Path

import polars as pl
import polars_corpus as plc
import pytest
from polars_corpus import read_text_corpus, scan_text_corpus
from polars_corpus.corpus_io import TextCorpusReader

SAMPLE = """The/DT quick/JJ brown/NN fox/NN
jumps/VBZ over/IN the/DT lazy/JJ dog/NN

Another/DT sentence/NN here/RB
"""


@pytest.fixture
def write_corpus(tmp_path: Path):
    """Write each string to its own file and hand back the paths."""

    def write(*contents: str) -> list[Path]:
        paths = []
        for i, content in enumerate(contents):
            path = tmp_path / f"corpus{i}.txt"
            path.write_text(content)
            paths.append(path)
        return paths

    return write


@pytest.fixture
def sample_file(write_corpus):
    return write_corpus(SAMPLE)[0]


@pytest.mark.parametrize(
    "load", [read_text_corpus, lambda p: scan_text_corpus(p).collect()]
)
def test_read_text_corpus(load, sample_file):
    """The eager and lazy entry points produce the same frame."""
    df = load([sample_file])

    assert df.columns == ["token", "pos", "sentence_tag", "file_id"]
    # Three blank-line-delimited sentences.
    assert df.filter(pl.col("sentence_tag") == "B").height == 3
    # Every token carries the path it was read from.
    assert set(df["file_id"]) == {str(sample_file)}


def test_token_tag_parsing(sample_file):
    rows = list(TextCorpusReader([sample_file]).read_file(sample_file))

    assert rows[0] == {"token": "The", "pos": "DT", "sentence_tag": "B"}
    assert rows[1]["sentence_tag"] == "I"


def test_scan_pushes_down_predicate_and_projection(sample_file):
    df = (
        scan_text_corpus([sample_file])
        .select("token")
        .filter(pl.col("token").str.contains("fox|dog"))
        .collect()
    )

    assert df.columns == ["token"]
    assert set(df["token"]) == {"fox", "dog"}


def test_malformed_line_raises(write_corpus):
    """A token with no /tag is an error, not a silently dropped row."""
    (path,) = write_corpus("Good/JJ morning/NN everyone\nMalformedLine noSlashHere\n")

    with pytest.raises(ValueError):
        list(TextCorpusReader([path]).read_file(path))


def test_multiple_files_combined(write_corpus):
    paths = write_corpus(
        "First/DT file/NN\n", "Second/DT one/CD\nThird/JJ one/NN too/RB\n"
    )
    df = read_text_corpus(paths)

    assert df.filter(pl.col("token") == "First").height == 1
    assert df.filter(pl.col("token") == "Third").height == 1
    # Each file's tokens carry its own id, contiguously.
    assert df["file_id"].to_list() == [str(paths[0])] * 2 + [str(paths[1])] * 5


def test_empty_file(write_corpus):
    assert read_text_corpus(write_corpus("")).is_empty()


def test_output_is_searchable_with_default_columns(sample_file):
    """The tag column must be named `pos`, which is what search() looks for."""
    results = plc.search(read_text_corpus([sample_file]), "the _JJ _NN")
    assert results is not None and len(results._matches) > 0
