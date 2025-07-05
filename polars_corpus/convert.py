from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator

import polars as pl

if TYPE_CHECKING:
    from nltk.corpus.reader.api import CorpusReader

__all__ = ["from_nltk"]


def from_nltk(corpus: CorpusReader) -> pl.DataFrame:
    """Converts an NLTK corpus into a Polars DataFrame with automatic detection of corpus features.

    Parameters
    ----------
    corpus : CorpusReader
        Any NLTK CorpusReader object (e.g., PlaintextCorpusReader, CategorizedPlaintextCorpusReader,
        TaggedCorpusReader)

    Returns
    -------
    DataFrame
        A Polars DataFrame where each row represents a token with associated metadata.

    Notes
    -----
    The resulting DataFrame contains the following columns (depending on corpus type):

    * token (str): The word/token text
    * tag (str, optional): Part-of-speech tag (if corpus provides tagged data)
    * sentence_tag (str, optional): Sentence boundary marker ("B" for beginning, "I" for inside)
    * file_id (str): Source file identifier
    * category (str, optional): Corpus category (if corpus is categorized)
"""
    category_dict = dict()
    if hasattr(corpus, "categories"):
        for category in corpus.categories():
            for file_id in corpus.fileids(category):
                category_dict[file_id] = category
    corpus_data = []
    for file_id in corpus.fileids():
        for token_dict in convert_file(corpus, file_id):
            token_dict["file_id"] = file_id
            if file_id in category_dict:
                token_dict["category"] = category_dict[file_id]
            corpus_data.append(token_dict)
    return pl.DataFrame(corpus_data)


def convert_file(corpus: CorpusReader, file_id: str) -> Generator[dict[str, str]]:
    if hasattr(corpus, "sents"):
        if hasattr(corpus, "tagged_sents"):
            sentences = corpus.tagged_sents(file_id)
        else:
            sentences = corpus.sents(file_id)
        for sentence in sentences:
            first_word = True
            for token_dict in convert_token(sentence):
                if first_word:
                    token_dict["sentence_tag"] = "B"
                    first_word = False
                else:
                    token_dict["sentence_tag"] = "I"
                yield token_dict
    else:
        if hasattr(corpus, "tagged_words"):
            tokens = corpus.tagged_words(file_id)
        else:
            tokens = corpus.words(file_id)
        for token_dict in convert_token(tokens):
            yield token_dict


def convert_token(tokens: list[Any]) -> Generator[dict[str, str]]:
    for token in tokens:
        token_dict = {}
        if type(token) is tuple:
            token_dict["token"] = token[0]
            token_dict["tag"] = token[1]
        else:
            token_dict["token"] = token
        yield token_dict
