from __future__ import annotations

from typing import Optional

import polars as pl
from nltk.corpus.reader.api import CorpusReader
from collections import defaultdict

__all__ = ["from_nltk"]

def from_nltk(corpus: CorpusReader) -> dict:
    corpus_data = defaultdict(list)
    for category in get_categories(corpus):
        for file_id in get_file_ids(corpus, category):
            for sentence in get_sentences(corpus, file_id):
                first_word = True
                for token in sentence:
                    if type(token) == tuple:
                        corpus_data['token'].append(token[0])
                        corpus_data['tag'].append(token[1])
                    else:
                        corpus_data['token'].append(token[0])
                    if category is not None:
                        corpus_data['category'].append(category)
                    if hasattr(corpus, "sents"):
                        if first_word:
                            corpus_data['sentence_tag'].append('B')
                            first_word = False
                        else:

                            corpus_data['sentence_tag'].append('I')
                    corpus_data['file_id'].append(file_id)
    return pl.DataFrame(corpus_data)

def get_categories(corpus: nltk.corpus.reader.api.CorpusReader) -> Optional[list]:
    if hasattr(corpus, 'categories'):
        return corpus.categories()
    else:
        return [None]

def get_file_ids(corpus: nltk.corpus.reader.api.CorpusReader, category=Optional[str]) -> list:
    if category is None:
        return corpus.fileids()
    else:
        return corpus.fileids(category)

def get_sentences(corpus: nltk.corpus.reader.api.CorpusReader, file_id: str) -> Optional[list]:
    if hasattr(corpus, 'tagged_sents'):
        return [sent for sent in corpus.tagged_sents(fileids=file_id)]
    elif hasattr(corpus, 'sents'):
        return [sent for sent in corpus.sents(fileids=file_id)]
    elif hasattr(corpus, 'tagged_words'):
        return [corpus.tagged_words(fileids=file_id)]
    elif hasattr(corpus, 'words'):
        return [corpus.words(fileids=file_id)]





