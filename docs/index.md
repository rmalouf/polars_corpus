# polars-corpus

A hybrid Python/Rust extension for Polars providing corpus search, concordancing, and statistical measures. Designed to comfortably handle 100M+ word corpora on 16GB memory.

## Features

- **Corpus search** using Simple (BNCweb-style) or CQP query syntax
- **Concordancing** with KWIC output
- **Statistical measures** for collocation and frequency analysis
- **Polars integration** via the `.corpus` namespace

## Installation

```bash
pip install polars-corpus
```

## Quick Start

```python
import polars as pl
import polars_corpus

df = pl.read_parquet("corpus.parquet")
df.corpus.search("[word='run' %c]")
```
