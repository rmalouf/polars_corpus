from pathlib import Path

import polars as pl
import polars_corpus as plc
import pytest
from polars_corpus import read_wlp_corpus, scan_wlp_corpus
from polars_corpus.corpus_io import WlpCorpusReader

COLUMNS = ["token", "lemma", "pos", "file_id"]

# Two texts, in the shape Mark Davies's COCA/COHA .wlp files take: a "##" line
# giving the text id, then one token per line as word, lemma and tag.
SAMPLE = """##4000161\t\t
Section\tsection\tnn1
:\t:\ty
Life\tlife\tnn1
and\tand\tcc
Letters\tletter\tnn2
##4000162\t\t
the\tthe\tat
quick\tquick\tjj
brown\tbrown\tjj
fox\tfox\tnn1
@\t\tii
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
    "load", [read_wlp_corpus, lambda p: scan_wlp_corpus(p).collect()]
)
def test_read_wlp_corpus(load, sample_file):
    """The eager and lazy entry points produce the same frame."""
    df = load([sample_file])

    assert df.columns == COLUMNS
    assert df.height == 10
    assert df.row(0) == ("Section", "section", "nn1", "4000161")
    # An empty lemma is a value, not a short row.
    assert df.row(-1) == ("@", "", "ii", "4000162")


@pytest.mark.parametrize(
    "load", [read_wlp_corpus, lambda p: scan_wlp_corpus(p).collect()]
)
def test_file_id_comes_from_the_text_header(load, sample_file):
    """A "##" line names the text, so one file holds many file ids."""
    df = load([sample_file])

    assert df["file_id"].to_list() == ["4000161"] * 5 + ["4000162"] * 5


def test_token_lemma_pos_parsing(sample_file):
    rows = list(WlpCorpusReader([sample_file]).read_file(sample_file))

    assert rows[0] == {
        "token": "Section",
        "lemma": "section",
        "pos": "nn1",
        "file_id": "4000161",
    }


def test_scan_pushes_down_predicate_and_projection(sample_file):
    df = (
        scan_wlp_corpus([sample_file])
        .select("lemma")
        .filter(pl.col("lemma").str.contains("fox|letter"))
        .collect()
    )

    assert df.columns == ["lemma"]
    assert set(df["lemma"]) == {"fox", "letter"}


def test_multiple_files_combined(write_corpus):
    paths = write_corpus("##1\t\t\nFirst\tfirst\tmd\n", SAMPLE)
    df = read_wlp_corpus(paths)

    assert df.height == 11
    # Text ids, not paths, and each file's tokens stay contiguous.
    assert df["file_id"].to_list() == ["1"] + ["4000161"] * 5 + ["4000162"] * 5


def test_empty_file(write_corpus):
    """Eager reading of an empty corpus gives an empty frame."""
    assert read_wlp_corpus(write_corpus("")).is_empty()


def test_scan_schema_is_fixed(write_corpus):
    """The lazy schema is declared up front, so it holds with no tokens read."""
    lf = scan_wlp_corpus(write_corpus(""))

    assert lf.collect_schema() == pl.Schema(dict.fromkeys(COLUMNS, pl.String))
    assert lf.collect().columns == COLUMNS


def test_undecodable_bytes_are_replaced(tmp_path: Path):
    """COCA files carry stray bytes; they must not stop the read."""
    path = tmp_path / "bad.txt"
    path.write_bytes(b"##1\t\t\nca\xf1on\tcanyon\tnn1\n")

    assert read_wlp_corpus([path])["token"].to_list() == ["ca\ufffdon"]


def test_output_is_searchable_with_default_columns(sample_file):
    """The columns must be named what search() looks for by default."""
    results = plc.search(read_wlp_corpus([sample_file]), "the _jj _jj _nn1")

    assert results is not None and len(results.matches) == 1
