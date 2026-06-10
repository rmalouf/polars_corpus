# Data Format

Corpora are Polars DataFrames stored as Parquet files. Each row is one token.

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `token` | `Utf8` | Surface form of the word |
| `pos` | `Utf8` | Part-of-speech tag |
| `mode` | `Utf8` | Register (`spoken`/`written`) |
| `fileid` | `Utf8` | Source document identifier |

## Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `c5` | `Utf8` | BNC C5 fine-grained POS tag |
| `lemma` | `Utf8` | Lemmatized form |
