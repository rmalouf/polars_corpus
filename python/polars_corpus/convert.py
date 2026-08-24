from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, cast

import polars as pl

if TYPE_CHECKING:
    from nltk.corpus.reader.api import CategorizedCorpusReader, CorpusReader

__all__ = ["from_nltk"]


def from_nltk(corpus: CorpusReader) -> pl.DataFrame:
    """
    Read an NLTK corpus into a Polars DataFrame, one row per token.

    Works with any corpus NLTK can read: Brown, the Gutenberg texts, anything
    under `nltk.corpus`, or a reader pointed at a directory of your own.

    The columns depend on what the reader offers. A tagged corpus gets a `pos`
    column, a corpus read as sentences gets a `sentence_tag` column, and a
    categorized corpus gets a `category` column.

    Parameters
    ----------
    corpus : CorpusReader
        Any NLTK corpus reader, e.g. `nltk.corpus.brown` or a
        `PlaintextCorpusReader` over a directory of your own.

    Returns
    -------
    pl.DataFrame
        One row per token, in corpus order, with as many of these columns as
        the reader can supply:

        - `token` : the word itself
        - `pos` : its part-of-speech tag, if the corpus is tagged
        - `sentence_tag` : "B" on the first token of each sentence and "I" on
          the rest, if the reader reads sentences
        - `file_id` : the file the token came from
        - `category` : the file's category, if the corpus is categorized

    Raises
    ------
    AttributeError
        If the reader exposes none of `sents()`, `tagged_words()` or
        `words()`, so its tokens cannot be read.

    Examples
    --------
    >>> import nltk
    >>> import polars_corpus as plc
    >>> brown = plc.from_nltk(nltk.corpus.brown)
    >>> # `category` is there because Brown is categorized:
    >>> brown.group_by("category").len()
    """
    category_dict = dict()
    if hasattr(corpus, "categories"):
        # CategorizedCorpusReader is a mixin, so a categorized corpus is only
        # identifiable by duck-typing. It supplies categories() and widens
        # fileids(), which on the plain reader takes no arguments.
        categorized = cast("CategorizedCorpusReader", corpus)
        for category in categorized.categories():
            for file_id in categorized.fileids(category):
                category_dict[file_id] = category
    corpus_data = []
    for file_id in corpus.fileids():
        for token_dict in _convert_file(corpus, file_id):
            token_dict["file_id"] = file_id
            if file_id in category_dict:
                token_dict["category"] = category_dict[file_id]
            corpus_data.append(token_dict)
    return pl.DataFrame(corpus_data)


def _convert_file(corpus: CorpusReader, file_id: str) -> Generator[dict[str, str]]:
    if hasattr(corpus, "sents"):
        if hasattr(corpus, "tagged_sents"):
            sentences = corpus.tagged_sents(file_id)
        else:
            sentences = corpus.sents(file_id)
        for sentence in sentences:
            first_word = True
            for token_dict in _convert_token(sentence):
                if first_word:
                    token_dict["sentence_tag"] = "B"
                    first_word = False
                else:
                    token_dict["sentence_tag"] = "I"
                yield token_dict
    else:
        if hasattr(corpus, "tagged_words"):
            tokens = corpus.tagged_words(file_id)
        elif hasattr(corpus, "words"):
            tokens = corpus.words(file_id)
        else:
            raise AttributeError(
                f"{type(corpus).__name__} exposes none of sents(), tagged_words(), "
                "or words(), so its tokens cannot be read"
            )
        for token_dict in _convert_token(tokens):
            yield token_dict


def _convert_token(tokens: list[Any]) -> Generator[dict[str, str]]:
    for token in tokens:
        token_dict = {}
        if type(token) is tuple:
            token_dict["token"] = token[0]
            token_dict["pos"] = token[1]
        else:
            token_dict["token"] = token
        yield token_dict
