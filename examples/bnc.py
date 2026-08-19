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
#
# Speaker metadata is denormalized onto every token rather than written to a
# second file to be joined back, so selecting speakers is a filter and not a
# join.  It is cheap to carry: a text has a handful of speakers, so the columns
# are long runs that Parquet dictionary-encodes away -- a few percent of a
# spoken text, and nothing at all for a written one, where they are all null.

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from lxml import etree

import pyarrow.parquet as pq

TEXTS = Path("/Volumes/Corpora/bnc_xml/Texts")
CORPUS_PATH = Path("bnc.parquet")
# Parsing runs in worker processes, not threads: lxml frees the GIL while it
# parses, but that is only a fifth of the work here -- walking the tree and
# building the DataFrame is Python, and threads measure no faster than serial.
N_WORKERS = 8

# Tokens to aim for per row group.  A row group is the unit a scan prunes to,
# so this is the granularity a lazy `slice` can seek at: smaller groups mean a
# concordance over a few texts decodes less around them, at the cost of more
# metadata and slightly worse compression.
ROW_GROUP_TOKENS = 250_000

# The <person> attributes worth keeping, as column -> (attribute, the code the
# BNC uses for "not recorded").  Those codes read back as nulls, so a null test
# covers both an unrecorded speaker and a written text with no speaker at all.
SPEAKER_ATTRS = {
    "sex": ("sex", "u"),
    "age_group": ("ageGroup", "X"),
    "soc": ("soc", "UU"),
    "dialect": ("dialect", "NONE"),
}
# The rest of the metadata is in child elements rather than attributes.  `age`
# and `educ` are left out as before: `educ` is "X" for all but a handful of
# speakers, and `age` is free text ("40", "30+") that `age_group` already bins.
SPEAKER_ELEMENTS = {
    "pers_name": "persName",
    "occupation": "occupation",
    "pers_note": "persNote",
}

# Everything stays String: Parquet dictionary-encodes the repetitive columns on
# disk, and Categorical would only cost memory once the corpus is read back.
TOKEN_SCHEMA = pl.Schema(
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

# Keyed on speaker_id, which the token side already carries, so joining the two
# gives exactly TOKEN_SCHEMA followed by the metadata columns.
SPEAKER_SCHEMA = pl.Schema(
    {"speaker_id": pl.String}
    | {column: pl.String for column in SPEAKER_ATTRS | SPEAKER_ELEMENTS}
)

CORPUS_SCHEMA = pl.Schema(
    TOKEN_SCHEMA | {c: t for c, t in SPEAKER_SCHEMA.items() if c != "speaker_id"}
)


def get_speakers(doc):
    """The document's participants, one row per speaker."""
    rows = []
    for person in doc.xpath("//person"):
        row = [person.get("{http://www.w3.org/XML/1998/namespace}id")]
        row += [
            None if (value := person.get(attr)) == absent else value
            for attr, absent in SPEAKER_ATTRS.values()
        ]
        row += [
            found[0].text.strip() if (found := person.xpath(tag)) else None
            for tag in SPEAKER_ELEMENTS.values()
        ]
        rows.append(row)
    return pl.DataFrame(rows, schema=SPEAKER_SCHEMA, orient="row")


def get_xml(filename):
    doc = etree.parse(str(filename))
    docid = doc.xpath('//idno[@type="bnc"]')[0].text
    # G3C.xml is an earlier copy of HWX.xml, header and all, so two files
    # claim the id HWX, which would give that id two runs of tokens far apart
    # -- exactly what the lazy search refuses to search.  The stray copy is
    # the one not named for the id it carries, so dropping it here keeps
    # HWX.xml, the corrected December 2006 text, and needs no pass over the
    # other files to work out which of the two to prefer.
    if docid != Path(filename).stem:
        return None
    if text := doc.xpath("//wtext"):
        text_mode = "written"
        text_type = text[0].xpath("./@type")[0]
    elif text := doc.xpath("//stext"):
        text_mode = "spoken"
        text_type = text[0].xpath("./@type")[0]
    else:
        raise ValueError("Unknown text type")

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

    # Join rather than widen the loop above: the metadata is per speaker, and
    # appending seven more constants per token would cost more than the join.
    return pl.DataFrame(data, schema=TOKEN_SCHEMA).join(
        get_speakers(doc), on="speaker_id", how="left", maintain_order="left"
    )


def write_row_group(writer, batch):
    """Write a batch of whole texts as one row group."""
    table = pl.concat(batch).to_arrow()
    # Without the explicit size pyarrow caps a row group at 1M rows, which
    # would put the seam somewhere inside a text rather than between two.
    writer.write_table(table, row_group_size=table.num_rows)


def convert(paths, corpus_path=CORPUS_PATH):
    """Parse `paths` in order, appending each batch of texts to the Parquet file."""
    schema = pl.DataFrame(schema=CORPUS_SCHEMA).to_arrow().schema
    batch, batch_tokens = [], 0
    with (
        pq.ParquetWriter(corpus_path, schema, compression="zstd") as writer,
        ProcessPoolExecutor(N_WORKERS) as pool,
    ):
        # `map` hands the texts back in the order the paths were given, which
        # is what keeps each file_id in a single run, and `buffersize` caps
        # how far the workers may run ahead of the writer -- without it every
        # text parsed so far would be held in memory, which is the one thing
        # this script is not allowed to do.
        for done, corpus_df in enumerate(
            pool.map(get_xml, paths, buffersize=2 * N_WORKERS), start=1
        ):
            if corpus_df is None:  # a stray copy of another text; see get_xml
                continue
            # Close the group before the text that would overrun it, so only
            # a text longer than the target lands in one alone.
            if batch and batch_tokens + corpus_df.height > ROW_GROUP_TOKENS:
                write_row_group(writer, batch)
                batch, batch_tokens = [], 0
            batch.append(corpus_df)
            batch_tokens += corpus_df.height
            print(f"\r{done:,} / {len(paths):,} texts", end="", flush=True)
        if batch:
            write_row_group(writer, batch)
    print()


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

# Searching it back never loads the corpus, and the speaker metadata rides
# along on the results:
#
#     import polars as pl
#     import polars_corpus  # noqa: F401
#
#     corpus = pl.scan_parquet("bnc.parquet")
#     results = corpus.corpus.search("{take} * for granted")
#     print(results.concordance())
