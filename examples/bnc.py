## Convert XML BNC into a parquet file polars can scan lazily

# BNC XML to Parquet
#
# This script reads BNC XML files and converts them to a Parquet file using Polars.  The BNC XML files are available from the BNC website: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554
# [Reference Guide for the British National Corpus (XML Edition)](http://www.natcorp.ox.ac.uk/docs/URG/index.html)
#
# The BNC is larger than memory on the machines this is meant for, so both ends
# of the pipeline stream: texts are appended to the Parquet file as they are
# parsed, never gathered into one DataFrame first, and the file is laid out for
# the lazy search path in `matcher.py`, which asks for
#
#   * a `file_id` column, since that is what it chunks the corpus by;
#   * every token of a text in one contiguous run -- so texts go in whole, one
#     at a time, in sorted order;
#   * row groups small enough that slicing to a handful of texts doesn't decode
#     much else, and aligned to text boundaries so it decodes nothing partial.

from collections import defaultdict
from pathlib import Path

import polars as pl
from lxml import etree

from joblib import Parallel, delayed
from joblib_progress import joblib_progress
import pyarrow.parquet as pq

TEXTS = Path("/Volumes/Corpora/bnc_xml/Texts")
CORPUS_PATH = Path("bnc.parquet")
SPEAKERS_PATH = Path("bnc-speakers.parquet")
N_JOBS = 8

# Tokens to aim for per row group.  A row group is the unit a scan prunes to,
# so this is the granularity a lazy `slice` can seek at: smaller groups mean a
# concordance over a few texts decodes less around them, at the cost of more
# metadata and slightly worse compression.
ROW_GROUP_TOKENS = 250_000

# Everything stays String: Parquet dictionary-encodes the repetitive columns on
# disk, and Categorical would only cost memory once the corpus is read back.
CORPUS_SCHEMA = pl.Schema(
    {
        "token": pl.String,
        "lemma": pl.String,
        "pos": pl.String,
        "tag": pl.String,
        "sentence_tag": pl.String,
        "mode": pl.String,
        "text_type": pl.String,
        "file_id": pl.String,
        "speaker_id": pl.String,
    }
)

SPEAKERS_SCHEMA = pl.Schema(
    {
        "speaker_id": pl.String,
        "sex": pl.String,
        "ageGroup": pl.String,
        "dialect": pl.String,
        #'educ': pl.String,
        "soc": pl.String,
        "persName": pl.String,
        #'age': pl.String,
        "occupation": pl.String,
        "persNote": pl.String,
    }
)


def get_xml(filename):
    doc = etree.parse(str(filename))
    # The id comes from the file name, not from the <idno> inside: G3C.xml is
    # a copy of HWX.xml, idno and all, so reading the header would give one
    # file id two runs of tokens far apart -- exactly what the lazy search
    # refuses to search.
    docid = Path(filename).stem
    if text := doc.xpath("//wtext"):
        text_mode = "written"
        text_type = text[0].xpath("./@type")[0]
    elif text := doc.xpath("//stext"):
        text_mode = "spoken"
        text_type = text[0].xpath("./@type")[0]
    else:
        raise ValueError("Unknown text type")

    speakers = defaultdict(list)
    for person in doc.xpath("//person"):
        speakers["speaker_id"].append(
            person.get("{http://www.w3.org/XML/1998/namespace}id")
        )
        for attr in [
            "ageGroup",
            "soc",
            "sex",
            "persName",
            "occupation",
            "dialect",
            "persNote",
        ]:
            if (t := person.get(attr)) is None:
                speakers[attr].append(None)
            else:
                speakers[attr].append(t)

    speakers_df = pl.DataFrame(speakers, schema=SPEAKERS_SCHEMA)

    data = defaultdict(list)
    for s in doc.xpath("//s"):
        if u := s.xpath("ancestor::u"):
            speaker_id = u[0].get("who")
        else:
            speaker_id = None
        sent_tag = "B"
        for token in s.xpath(
            ".//w | .//c | .//gap | .//unclear | .//pause | .//vocal | .//event"
        ):
            if (token.tag == "c" or token.tag == "w") and token.text is None:
                pass
            else:
                data["speaker_id"].append(speaker_id)
                data["mode"].append(text_mode)
                data["text_type"].append(text_type)
                data["file_id"].append(docid)
                data["sentence_tag"].append(sent_tag)
                if token.tag == "w":
                    data["token"].append(token.text.strip())
                    data["tag"].append(token.get("c5"))
                    data["lemma"].append(token.get("hw"))
                    data["pos"].append(token.get("pos"))
                elif token.tag == "c":
                    data["token"].append(token.text.strip())
                    data["tag"].append(token.get("c5"))
                    data["lemma"].append(None)
                    data["pos"].append("STOP")
                else:
                    data["token"].append(f"<{token.tag}/>")
                    data["tag"].append(None)
                    data["lemma"].append(None)
                    data["pos"].append(None)
            sent_tag = "I"

    corpus_df = pl.DataFrame(data, schema=CORPUS_SCHEMA)
    return corpus_df, speakers_df


def write_row_group(writer, batch):
    """Write a batch of whole texts as one row group."""
    table = pl.concat(batch).to_arrow()
    # Without the explicit size pyarrow caps a row group at 1M rows, which
    # would put the seam somewhere inside a text rather than between two.
    writer.write_table(table, row_group_size=table.num_rows)


def convert(paths, corpus_path=CORPUS_PATH, speakers_path=SPEAKERS_PATH):
    """Parse `paths` in order, appending each batch of texts to the Parquet file."""
    schema = pl.DataFrame(schema=CORPUS_SCHEMA).to_arrow().schema
    batch, batch_tokens = [], 0
    speakers = []
    with pq.ParquetWriter(corpus_path, schema, compression="zstd") as writer:
        with joblib_progress(total=len(paths)):
            # A generator holds only the texts in flight, and joblib yields
            # them back in the order the paths were given -- which is what
            # keeps each file_id in a single run.
            with Parallel(n_jobs=N_JOBS, return_as="generator", verbose=0) as parallel:
                for corpus_df, speakers_df in parallel(
                    delayed(get_xml)(path) for path in paths
                ):
                    speakers.append(speakers_df)  # a few thousand rows in all
                    # Close the group before the text that would overrun it,
                    # so only a text longer than the target lands in one alone.
                    if batch and batch_tokens + corpus_df.height > ROW_GROUP_TOKENS:
                        write_row_group(writer, batch)
                        batch, batch_tokens = [], 0
                    batch.append(corpus_df)
                    batch_tokens += corpus_df.height
        if batch:
            write_row_group(writer, batch)

    pl.concat(speakers).write_parquet(speakers_path)


def check_corpus(corpus_path=CORPUS_PATH, file_id_column="file_id"):
    """Check the Parquet file the way a lazy search will read it."""
    tokens, files, runs = (
        pl.scan_parquet(corpus_path)
        .select(
            pl.len().alias("tokens"),
            pl.col(file_id_column).n_unique().alias("files"),
            (pl.col(file_id_column).rle_id().max() + 1).alias("runs"),
        )
        .collect(engine="streaming")
        .row(0)
    )
    if runs != files:
        raise ValueError(
            f"{corpus_path} interleaves its texts: {runs:,} runs of "
            f"{file_id_column} for {files:,} distinct values"
        )
    print(f"{corpus_path}: {tokens:,} tokens in {files:,} texts")


if __name__ == "__main__":
    convert(sorted(TEXTS.glob("**/*.xml")))
    check_corpus()

# Searching it back never loads the corpus:
#
#     import polars as pl
#     import polars_corpus  # noqa: F401
#
#     corpus = pl.scan_parquet("bnc.parquet")
#     results = corpus.corpus.search("{take} * for granted")
#     print(results.concordance())
