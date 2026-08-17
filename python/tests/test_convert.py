import polars as pl
import polars_corpus as plc
import pytest
from nltk.corpus import brown, lin_thesaurus, state_union, swadesh207, treebank
from polars_corpus.convert import from_nltk

# Each reader exposes a different subset of annotations, and from_nltk emits a
# column only where the underlying reader supplies one.
NLTK_CORPORA = [
    pytest.param(
        brown,
        ["token", "pos", "sentence_tag", "file_id", "category"],
        ("race", "NN", "I", "ca01", "news"),
        id="brown",
    ),
    pytest.param(
        treebank,
        ["token", "pos", "sentence_tag", "file_id"],
        ("be", "VB", "I", "wsj_0004.mrg"),
        id="treebank",
    ),
    pytest.param(
        state_union,
        ["token", "sentence_tag", "file_id"],
        ("and", "I", "1945-Truman.txt"),
        id="state_union",
    ),
    pytest.param(
        swadesh207,
        ["token", "file_id"],
        ("ñahn\tɲâhǹ", "swadesh207/adj-000.txt"),
        id="swadesh207",
    ),
]


@pytest.mark.parametrize("reader,columns,row_1000", NLTK_CORPORA)
def test_from_nltk(reader, columns, row_1000):
    c = from_nltk(reader)

    assert c.columns == columns
    assert c.get_column("file_id").n_unique() == len(reader.fileids())
    assert c.row(1000) == row_1000

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


def test_unsupported_reader():
    """A reader with no words() interface cannot be converted."""
    with pytest.raises(AttributeError):
        from_nltk(lin_thesaurus)
