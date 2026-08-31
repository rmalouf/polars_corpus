from pathlib import Path
import polars as pl
import polars_corpus as plc

corpus = plc.scan_wlp_corpus(sorted(Path("/Volumes/Corpora/COCA/wlp").glob("wlp*.zst")))
corpus.sink_parquet('coha.parquet')

