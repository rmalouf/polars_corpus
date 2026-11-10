# polars-corpus

polars-corpus is a Polars extensions for corpus linguistics workflow: search,
concordances, frequency lists, collocations, keywords, dispersion, and lexical
diversity. It handles large corpora (100M+ words) on modest hardware without
external servers or databases.

A corpus is a DataFrame (or LazyFrame) with one row per token. The default
column names are `token`, `pos`, `lemma` and `file_id`; every function that
reads one of these takes a `*_column` or `expr` argument to point it elsewhere.
A LazyFrame is searched a chunk of files at a time, so the corpus does not have
to fit in memory.

## Installation

```bash
pip install polars-corpus
```

The plotting functions need matplotlib, which is in the `examples` extra:

```bash
pip install "polars-corpus[examples]"
```

Requires Python 3.11 or later and Polars 1.36 or later, including 2.0.

## Quick start

```python
import nltk
import polars as pl
import polars_corpus as plc

nltk.download("brown")
corpus = plc.from_nltk(nltk.corpus.brown)

# Simple (BNCweb-style) query: "small" followed by a noun
plc.search(corpus, "small _{SUBST}").concordance("token", window=5)

# CQP query
plc.search_cqp(corpus, '[pos="JJ"] [token="(?i)house"]').concordance(window=2)

word = pl.col("token").str.to_lowercase()
plc.frequency_list(corpus, word)
plc.collocations(plc.search(corpus, "house"), word, "logdice")
```

The same functions are available through a `.corpus` namespace on DataFrames,
LazyFrames and expressions:

```python
corpus.corpus.search("america").concordance("token", window=2)
```

## Examples

- [Word frequencies](notebooks/frequencies.ipynb)
- [Concordances](notebooks/concordance.ipynb)
- [Collocations](notebooks/collocation.ipynb)
- [Keyword analysis](notebooks/keywords.ipynb)
- [Dispersion](notebooks/dispersion.ipynb)
- [Vocabulary](notebooks/vocabulary.ipynb)

## Reference

- [Reading corpora](io.md)
- [Frequency lists](frequencies.md)
- [Search](search.md)
- [Association metrics](assoc.md)
- [Collocation](collocations.md)
- [Lexical metrics](lexical.md)
- [Lexical dispersion](dispersion.md)
- [Keyword analysis](keywords.md)
- [Embeddings](embed.md)
- [Utilities](utils.md)
- [Simple query language](simple_query.md)
- [CQP query language](cqp_query.md)
