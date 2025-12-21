import polars as pl
import pytest
from nltk.corpus import brown, lin_thesaurus, state_union, swadesh207, treebank
from polars_corpus.convert import from_nltk


class TestNltkCorpora:
    def test_brown(self):
        c = from_nltk(brown)
        assert c.height == len([word for sent in brown.sents() for word in sent])
        assert c.get_column("file_id").n_unique() == len(brown.fileids())
        assert c.get_column("category").n_unique() == len(brown.categories())
        assert c.filter(pl.col("category") == "science_fiction").get_column(
            "file_id"
        ).n_unique() == len(brown.fileids("science_fiction"))
        assert c.filter(pl.col("sentence_tag") == "B").height == len(brown.sents())
        assert c.row(1000) == ("race", "NN", "I", "ca01", "news")

    def test_treebank(self):
        c = from_nltk(treebank)
        assert c.height == len([word for sent in treebank.sents() for word in sent])
        assert c.get_column("file_id").n_unique() == len(treebank.fileids())
        assert "category" not in c.columns
        assert c.filter(pl.col("sentence_tag") == "B").height == len(treebank.sents())
        assert c.row(1000) == ("be", "VB", "I", "wsj_0004.mrg")

    def test_state_union(self):
        c = from_nltk(state_union)
        assert c.height == len([word for sent in state_union.sents() for word in sent])
        assert c.get_column("file_id").n_unique() == len(state_union.fileids())
        assert "category" not in c.columns
        assert "tag" not in c.columns
        assert c.filter(pl.col("sentence_tag") == "B").height == len(
            state_union.sents()
        )
        assert c.row(1000) == ("and", "I", "1945-Truman.txt")

    def test_swadesh207(self):
        c = from_nltk(swadesh207)
        assert c.height == len(swadesh207.words())
        assert c.get_column("file_id").n_unique() == len(swadesh207.fileids())
        assert "category" not in c.columns
        assert "tag" not in c.columns
        assert "sentence_tag" not in c.columns
        assert c.row(1000) == ("ñahn\tɲâhǹ", "swadesh207/adj-000.txt")

    def test_lin_thesaurus(self):
        with pytest.raises(AttributeError):
            _ = from_nltk(lin_thesaurus)
