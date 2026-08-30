import polars as pl
import polars_corpus as plc
import pytest
from nltk.corpus import brown
from nltk.corpus.reader import (
    CorpusReader,
    PlaintextCorpusReader,
    TaggedCorpusReader,
    WordListCorpusReader,
)
from nltk.tokenize import LineTokenizer
from polars_corpus.convert import from_nltk

# from_nltk emits a column only where the reader supplies one, so what a case
# has to vary is the reader's interface, not the text. Only the categorized
# case needs a downloaded corpus; the rest are reader classes over tmp_path,
# which keeps CI to one nltk package. LineTokenizer is there to keep the
# plaintext reader off punkt, whose data would be another download.
PLAIN = "the quick brown fox\njumps over the lazy dog\n"
TAGGED = "the/DT quick/JJ brown/JJ fox/NN\njumps/VBZ over/IN the/DT lazy/JJ dog/NN\n"
WORDS = "alpha\nbeta\ngamma\n"

NLTK_CORPORA = [
    pytest.param(
        lambda root: brown,
        ["token", "pos", "sentence_tag", "file_id", "category"],
        (1000, ("race", "NN", "I", "ca01", "news")),
        id="brown",
    ),
    pytest.param(
        lambda root: TaggedCorpusReader(root, r".*\.pos"),
        ["token", "pos", "sentence_tag", "file_id"],
        (0, ("the", "DT", "B", "a.pos")),
        id="tagged",
    ),
    pytest.param(
        lambda root: PlaintextCorpusReader(
            root, r".*\.txt", sent_tokenizer=LineTokenizer()
        ),
        ["token", "sentence_tag", "file_id"],
        (0, ("the", "B", "a.txt")),
        id="plaintext",
    ),
    pytest.param(
        lambda root: WordListCorpusReader(root, r".*\.words"),
        ["token", "file_id"],
        (0, ("alpha", "a.words")),
        id="wordlist",
    ),
]


@pytest.fixture
def root(tmp_path):
    """A corpus directory holding one file per reader below."""
    (tmp_path / "a.txt").write_text(PLAIN)
    (tmp_path / "a.pos").write_text(TAGGED)
    (tmp_path / "a.words").write_text(WORDS)
    return str(tmp_path)


@pytest.mark.parametrize("reader,columns,row", NLTK_CORPORA)
def test_from_nltk(reader, columns, row, root):
    reader = reader(root)
    c = from_nltk(reader)
    index, expected = row

    assert c.columns == columns
    assert c.get_column("file_id").n_unique() == len(reader.fileids())
    assert c.row(index) == expected

    if "sentence_tag" in columns:
        assert c.height == sum(len(sent) for sent in reader.sents())
        assert c.filter(pl.col("sentence_tag") == "B").height == len(reader.sents())
    else:
        assert c.height == len(reader.words())


def test_categories_partition_the_files():
    c = from_nltk(brown)
    n_sf_files = c.filter(pl.col("category") == "science_fiction")["file_id"].n_unique()
    assert n_sf_files == len(brown.fileids("science_fiction"))


def test_output_is_searchable_with_default_columns():
    """The tag column must be named `pos`, which is what search() looks for."""
    c = from_nltk(brown).head(2000)
    results = plc.search(c, "the _JJ _NN")
    assert results is not None and len(results.matches) > 0


def test_unsupported_reader(root):
    """A reader with no words() interface cannot be converted."""
    with pytest.raises(AttributeError):
        from_nltk(CorpusReader(root, r".*\.txt"))
