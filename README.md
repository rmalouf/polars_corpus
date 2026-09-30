# polars-corpus

Corpus linguistics functions for Polars: search, concordances, frequency
lists, collocations, keywords, dispersion, and lexical diversity. The query
matcher is written in Rust; the rest is Python over Polars expressions.

A corpus is a DataFrame (or LazyFrame) with one row per token. The default
column names are `token`, `pos`, `lemma` and `file_id`; every function that
reads one of these takes a `*_column` or `expr` argument to point it elsewhere.
A LazyFrame is searched a chunk of files at a time, so the corpus does not have
to fit in memory.

The package is at version 0.2.0 and the API may still change.

Documentation: <https://rmalouf.github.io/polars_corpus/>

## Installation

```bash
pip install polars-corpus
```

The plotting functions need matplotlib, which is in the `examples` extra:

```bash
pip install "polars-corpus[examples]"
```

Requires Python 3.11 or later and Polars 1.36 or later (below 2.0).

## Usage

```python
import nltk
import polars as pl
import polars_corpus as plc

nltk.download("brown")
corpus = plc.from_nltk(nltk.corpus.brown)  # token, pos, sentence_tag, file_id, category

# Simple (BNCweb-style) query: "small" followed by a noun
results = plc.search(corpus, "small _{SUBST}")
results.concordance("token", window=5)

# CQP query
plc.search_cqp(corpus, '[pos="JJ"] [token="(?i)house"]').concordance(window=2)

word = pl.col("token").str.to_lowercase()

plc.frequency_list(corpus, word)
plc.collocations(plc.search(corpus, "house"), word, "logdice")
plc.keywords(
    corpus.filter(pl.col("category") == "news"),
    corpus.filter(pl.col("category") != "news"),
    word,
    "ll",
)
plc.dispersion(corpus, word, ["range", "dp"])
```

The same functions are available through a `.corpus` namespace on DataFrames,
LazyFrames and expressions:

```python
corpus.corpus.search("america").concordance("token", window=2)
```

## Contents

- **Search**: `search` (Simple query language) and `search_cqp` (CQP), with
  variable bindings in both. Results give KWIC concordances, hit distributions
  across a metadata column, and an interactive concordance viewer for Jupyter.
- **Frequency**: `frequency_list`, `ngrams`, `crosstab`.
- **Association measures**: PMI, MI3, log-Dice, t-score, z-score,
  log-likelihood, chi-squared, minimum sensitivity, simple maths, BIC, log
  ratio, %DIFF, odds ratio, Welch's t-test. Used by `collocations` and
  `keywords`.
- **Dispersion**: range, SD, CV, Juilland's D, Gries's DP, Burch's DA.
- **Lexical diversity**: TTR, MSTTR, MTLD, Yule's K, vocabulary growth,
  frequency spectrum.
- **Input**: plain text and WLP files (gzip or zstd compressed), NLTK corpus
  readers, and the BNC XML edition.
- **Plots**: `barcode_plot`, `dispersion_plot`, `keyword_plot`.

Not yet implemented: proximity operators (`<<3>>`, `<<s>>`) in the Simple query
language.

## Query languages

- Simple: `small _{SUBST}`, `( small | little ) _N*`. See
  [docs/simple_query.md](docs/simple_query.md).
- CQP: `[pos="JJ"] [lemma="house"]`. See [docs/cqp_query.md](docs/cqp_query.md).

## Examples

Worked notebooks are in [docs/notebooks/](docs/notebooks/) and rendered on the
documentation site: frequencies, concordances, collocations, keywords,
dispersion, and vocabulary.

## Development

Building from source needs a Rust toolchain and maturin:

```bash
make develop   # build the extension into the active environment
pytest
```

## License

MIT
