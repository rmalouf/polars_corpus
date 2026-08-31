import gzip
from pathlib import Path

import polars as pl
import polars_corpus as plc
import pytest
from polars.testing import assert_frame_equal
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


def test_read_frames_batches(sample_file):
    """The Rust reader hands back one frame per file, sliced on request."""
    reader = WlpCorpusReader([sample_file])

    assert [df.height for df in reader.read_frames()] == [10]
    assert [df.height for df in reader.read_frames(4)] == [4, 4, 2]


def test_scan_reads_only_the_rows_asked_for(write_corpus):
    """A limited query parses the batch it needs and stops, so a broken line
    further down the file is never reached."""
    (path,) = write_corpus(SAMPLE + "broken\tline\n")

    assert scan_wlp_corpus([path]).head(3).collect().height == 3
    # The same file read whole does hit the broken line.
    with pytest.raises(ValueError):
        read_wlp_corpus([path])


def test_malformed_line_raises(write_corpus):
    """A line that is not three tab-separated fields names itself."""
    (path,) = write_corpus("##1\t\t\nfine\tfine\tnn1\nbroken\tline\n")

    with pytest.raises(ValueError, match="corpus0.txt:3"):
        read_wlp_corpus([path])


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="nope.txt"):
        read_wlp_corpus([tmp_path / "nope.txt"])


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


def zstd_frame(data: bytes) -> bytes:
    """`data` wrapped in a zstd frame, stored as one uncompressed block.

    Built by hand because the standard library has no zstd before 3.14, and
    the reader's part is to recognize the frame and hand it to a decoder, not
    to compress anything. Frame header: single segment, four-byte content
    size. Block header: the size, then the raw and last-block flags.
    """
    header = b"\xa0" + len(data).to_bytes(4, "little")
    block = ((len(data) << 3) | 1).to_bytes(3, "little")
    return b"\x28\xb5\x2f\xfd" + header + block + data


# `bytes` leaves the sample alone, for the uncompressed case.
COMPRESSORS = {"plain": bytes, "gzip": gzip.compress, "zstd": zstd_frame}


@pytest.mark.parametrize("compress", COMPRESSORS.values(), ids=COMPRESSORS)
@pytest.mark.parametrize(
    "load", [read_wlp_corpus, lambda p: scan_wlp_corpus(p).collect()]
)
def test_compressed_corpus_reads_as_the_plain_one(
    load, compress, tmp_path, sample_file
):
    """gzip and zstd files are decoded on the way in.

    Every case is named ".gz", and the name decides nothing: the plain file is
    still read as text, and the zstd file is not read as gzip.
    """
    path = tmp_path / "compressed.wlp.gz"
    path.write_bytes(compress(SAMPLE.encode()))

    assert_frame_equal(load([path]), load([sample_file]))


def test_every_gzip_member_is_read(tmp_path: Path):
    """A `.gz` is often members concatenated, and all of them hold tokens."""
    head, tail = SAMPLE.split("##4000162")
    path = tmp_path / "concatenated.wlp.gz"
    path.write_bytes(
        gzip.compress(head.encode()) + gzip.compress(b"##4000162" + tail.encode())
    )

    assert read_wlp_corpus([path]).height == 10


def test_truncated_compressed_file_raises(tmp_path: Path):
    """A decoding error names the file, as a read error does."""
    path = tmp_path / "truncated.wlp.gz"
    path.write_bytes(gzip.compress(SAMPLE.encode())[:20])

    with pytest.raises(OSError, match="truncated.wlp.gz"):
        read_wlp_corpus([path])
