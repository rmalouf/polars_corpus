## convert BNC_XML corpus into a parquet file

from pathlib import Path
import polars_corpus as plc

BNC_ROOT = Path("/Volumes/Corpora/bnc_xml")

# Texts are parsed in worker processes, which re-import this module.
if __name__ == "__main__":
    plc.convert_bnc(BNC_ROOT, BNC_ROOT / "bnc.parquet")
