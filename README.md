# polars-corpus

Corpus linguistics tools for Polars DataFrames.

> **⚠️ Early Stage**: This project is under active development. The API and query language **will** change as we refine the design. Feedback and contributions are welcome!

## Installation

```bash
pip install polars-corpus
```

## Usage

```python
import polars as pl
import polars_corpus as plc

# Load your corpus as a DataFrame with token and POS columns
corpus = pl.read_parquet("brown.parquet")

# Search for patterns
results = plc.search(corpus, "small _{SUBST}", pos_column="tag")

# Generate concordances
results.concordance("token", window=5)

# Or use the .corpus namespace
corpus.corpus.search("america").concordance("token", window=2)
```

## Query Syntax

Supports two query languages:
- **Simple** (BNCweb-style): `small _{SUBST}`, `( small | little ) _N*`
- **CQP**: Standard CQP syntax for corpus queries

## Features

- Pattern search and concordancing
- Frequency distributions and collocations
- Keywords and statistical measures
- Handles 100M+ word corpora on 16GB memory

## Examples

See the `examples/` directory for notebooks covering:
- Concordance generation
- Collocation analysis
- Frequency analysis
- Keyword extraction
- Stylometry

## Requirements

- Python 3.11+
- Polars 1.35+

## License

MIT
