## convert COCA wlp corpus into a parquet file

from pathlib import Path
import polars as pl
import polars_corpus as plc

COCA_ROOT = Path("/Volumes/Corpora/COCA")

subgenres = pl.read_csv(
    COCA_ROOT / "subgenreCodes.txt", separator="\t", has_header=False
).rename({"column_1": "subGenre", "column_2": "text_type"})

meta = (
    pl.read_csv(
        COCA_ROOT / "coca-sources.txt",
        separator="\t",
        quote_char=None,
        encoding="utf8-lossy",
        skip_rows_after_header=1,
    )
    .rename({"textID": "file_id"})
    .join(subgenres, on="subGenre")
    .rename({"subGenre": "subgenre"})
    .with_columns(pl.col("file_id").cast(pl.String))
    .lazy()
)

corpus = (
    plc.scan_wlp_corpus(sorted(COCA_ROOT.glob("wlp/wlp*.zst")))
    .join(meta, on="file_id")
    .sink_parquet("coca.parquet")
)
