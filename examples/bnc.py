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
# Text and speaker metadata are denormalized onto every token rather than
# written to a second file to be joined back, so restricting a search to a
# subcorpus is a filter and not a join.  It is cheap to carry: a text column
# holds one value for the whole text and a speaker column changes only between
# turns, so both are long runs that Parquet dictionary-encodes away.

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

# The <person> attributes worth keeping, as column -> (attribute, the label for
# each code), decoded the way the text classifications above are so that a
# speaker column reads like the text columns beside it rather than in the BNC's
# codes.  The code for "not recorded" maps to None, so a null test covers both
# an unrecorded speaker and a written text with no speaker at all; a code no
# table names is kept as it stands, which is how `social_class` and `role` pass
# through the values there is no point in spelling out.  All five attributes are
# on all 6124 <person> elements, so nothing here has to cope with a missing one.
SPEAKER_ATTRS = {
    "sex": ("sex", {"m": "Male", "f": "Female", "u": None}),
    # The bands `author_age_group` and `respondent_age_group` use, so the three
    # are comparable.
    "age_group": (
        "ageGroup",
        {
            "Ag0": "0-14",
            "Ag1": "15-24",
            "Ag2": "25-34",
            "Ag3": "35-44",
            "Ag4": "45-59",
            "Ag5": "60+",
            "X": None,
        },
    ),
    # AB/C1/C2/DE are the grades the BNC's own tables use and what
    # `respondent_social_class` holds, so only the unknown code needs naming.
    "social_class": ("soc", {"UU": None}),
    "dialect": (
        "dialect",
        {
            "NONE": None,
            "CAN": "Canadian",
            "XDE": "German",
            "XEA": "East Anglian",
            "XFR": "French",
            "XHC": "Home Counties",
            "XHM": "Humberside",
            "XIR": "Irish",
            "XIS": "Indian subcontinent",
            "XLC": "Lancashire",
            "XLO": "London",
            "XMC": "Central Midlands",
            "XMD": "Merseyside",
            "XME": "North-east Midlands",
            "XMI": "Midlands",
            "XMS": "South Midlands",
            "XMW": "North-west Midlands",
            "XNC": "Central Northern England",
            "XNE": "North-east England",
            "XNO": "Northern England",
            "XOT": "Other or unidentifiable",
            "XSD": "Scottish",
            "XSL": "Lower south-west England",
            "XSS": "Central south-west England",
            "XSU": "Upper south-west England",
            "XUR": "European",
            "XUS": "American (US)",
            "XWA": "Welsh",
            "XWE": "West Indian",
        },
    ),
    # How the speaker stood to the respondent who carried the recorder, in a
    # demographically sampled text.  79 values, all of them already words, so
    # only the two that mean "not recorded" are named.
    "role": ("role", {"unspecified": None, "?": None}),
}
# The rest of the metadata is in child elements rather than attributes.  `age`
# and `educ` are left out as before: `educ` is "X" for all but a handful of
# speakers, and `age` is free text ("40", "30+") that `age_group` already bins.
SPEAKER_ELEMENTS = {
    "pers_name": "persName",
    "occupation": "occupation",
    "pers_note": "persNote",
}

# Each text carries one code from each of the classification taxonomies that
# apply to it, all run together in the `targets` attribute of its <catRef>.
# This maps column -> (taxonomy prefix, the label for each code), with the
# BNC's code for "not recorded" mapped to None: a null then covers both an
# unclassified text and a taxonomy that does not apply, so the written columns
# read back null throughout a spoken text and the spoken ones null throughout a
# written one.  Both the column names and the labels are the Reference Guide's,
# from its tables in section 1 (Design of the corpus), so a column here is
# searchable in the Guide by its own name.
#
# Three taxonomies are left out.  WRILEV (perceived difficulty) and WRISTA
# (estimated circulation) "were incorrectly differentiated during the
# preparation of the corpus and cannot be relied on" -- the Guide's own words.
# ALLTYP says nothing that `mode`, `text_type` and `written_medium` do not.
CATEGORIES = {
    "publication_date": (
        "ALLTIM",
        {0: None, 1: "1960-1974", 2: "1975-1984", 3: "1985-1993"},
    ),
    # Written texts.
    "written_domain": (
        "WRIDOM",
        {
            1: "Imaginative",
            2: "Informative: natural & pure science",
            3: "Informative: applied science",
            4: "Informative: social science",
            5: "Informative: world affairs",
            6: "Informative: commerce & finance",
            7: "Informative: arts",
            8: "Informative: belief & thought",
            9: "Informative: leisure",
        },
    ),
    "written_medium": (
        "WRIMED",
        {
            1: "Book",
            2: "Periodical",
            3: "Miscellaneous published",
            4: "Miscellaneous unpublished",
            5: "To-be-spoken",
        },
    ),
    "sampling_type": (
        "WRISAM",
        {
            0: None,
            1: "Whole text",
            2: "Beginning sample",
            3: "Middle sample",
            4: "End sample",
            5: "Composite sample",
        },
    ),
    "publication_place": (
        "WRIPP",
        {
            0: None,
            1: "UK (unspecific)",
            2: "Ireland",
            3: "UK: North",
            4: "UK: Midlands",
            5: "UK: South",
            6: "United States",
        },
    ),
    "author_type": (
        "WRIATY",
        {0: None, 1: "Corporate", 2: "Multiple", 3: "Sole"},
    ),
    "author_sex": ("WRIASE", {0: None, 1: "Male", 2: "Female", 3: "Mixed"}),
    "author_age_group": (
        "WRIAAG",
        {0: None, 1: "0-14", 2: "15-24", 3: "25-34", 4: "35-44", 5: "45-59", 6: "60+"},
    ),
    "author_domicile": (
        "WRIAD",
        {
            0: None,
            1: "UK and Ireland",
            2: "Commonwealth",
            3: "Continental Europe",
            4: "USA",
            5: "Elsewhere",
        },
    ),
    "audience_age": (
        "WRIAUD",
        {1: "Child", 2: "Teenager", 3: "Adult", 4: "Any"},
    ),
    "audience_sex": ("WRITAS", {0: None, 1: "Male", 2: "Female", 3: "Mixed"}),
    # Spoken texts, both samples.
    "region": ("SPOREG", {0: None, 1: "South", 2: "Midlands", 3: "North"}),
    "interaction_type": ("SPOLOG", {1: "Monologue", 2: "Dialogue"}),
    # Context-governed texts only.
    "spoken_context": (
        "SCGDOM",
        {
            1: "Educational/Informative",
            2: "Business",
            3: "Public/Institutional",
            4: "Leisure",
        },
    ),
    # Demographically sampled texts only: the recruit who carried the recorder,
    # not the speaker of the token, who is described by the speaker columns.
    "respondent_age_group": (
        "SDEAGE",
        {1: "0-14", 2: "15-24", 3: "25-34", 4: "35-44", 5: "45-59", 6: "60+"},
    ),
    "respondent_sex": ("SDESEX", {0: None, 1: "Male", 2: "Female"}),
    "respondent_social_class": (
        "SDECLA",
        {0: None, 1: "AB", 2: "C1", 3: "C2", 4: "DE"},
    ),
}

# Everything but the creation year stays String: Parquet dictionary-encodes the
# repetitive columns on disk, and Categorical would only cost memory once the
# corpus is read back.
TOKEN_SCHEMA = pl.Schema(
    {
        "token": pl.String,
        "lemma": pl.String,
        "pos": pl.String,
        "c5": pl.String,
        "sentence_tag": pl.String,
        "speaker_id": pl.String,
    }
)

# One value per text, so these are set as literals rather than appended a token
# at a time.  The two dates are different facts, not two readings of one:
# `creation_year` is <creation date>, "the year of original composition", while
# `publication_date` is the band the text was classified under, taken from the
# date of publication for a written text and of the recording for a spoken one.
# 383 texts are banded but composed in no recorded year, and 2 the other way
# round.
TEXT_SCHEMA = pl.Schema(
    {
        "file_id": pl.String,
        "mode": pl.String,
        "text_type": pl.String,
        "genre": pl.String,
        "creation_year": pl.Int16,
    }
    | {column: pl.String for column in CATEGORIES}
)

# Keyed on speaker_id, which the token side already carries, so joining it on
# is what completes CORPUS_SCHEMA.
SPEAKER_SCHEMA = pl.Schema(
    {"speaker_id": pl.String}
    | {column: pl.String for column in SPEAKER_ATTRS | SPEAKER_ELEMENTS}
)

CORPUS_SCHEMA = pl.Schema(
    TOKEN_SCHEMA
    | TEXT_SCHEMA
    | {c: t for c, t in SPEAKER_SCHEMA.items() if c != "speaker_id"}
)


def get_speakers(doc):
    """The document's participants, one row per speaker."""
    rows = []
    for person in doc.xpath("//person"):
        row = [person.get("{http://www.w3.org/XML/1998/namespace}id")]
        row += [
            labels.get(value := person.get(attr), value)
            for attr, labels in SPEAKER_ATTRS.values()
        ]
        row += [
            found[0].text.strip() if (found := person.xpath(tag)) else None
            for tag in SPEAKER_ELEMENTS.values()
        ]
        rows.append(row)
    return pl.DataFrame(rows, schema=SPEAKER_SCHEMA, orient="row")


def get_text_class(doc):
    """The document's classification codes, decoded to labels."""
    codes = doc.xpath("//catRef/@targets")[0].split()
    found = {}
    for column, (prefix, labels) in CATEGORIES.items():
        # A handful of texts are missing a code the rest of their sample has,
        # which reads back the same as one recorded as unknown.
        code = next((c for c in codes if c.startswith(prefix)), None)
        found[column] = labels[int(code.removeprefix(prefix))] if code else None
    return found


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
    elif text := doc.xpath("//stext"):
        text_mode = "spoken"
    else:
        raise ValueError("Unknown text type")

    # "0000" is what the header gives for a text whose date is unknown.
    created = doc.xpath("//creation/@date")[0]
    metadata = {
        "file_id": docid,
        "mode": text_mode,
        "text_type": text[0].get("type"),
        "genre": doc.xpath("//classCode")[0].text,
        "creation_year": None if created == "0000" else int(created),
    } | get_text_class(doc)

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
                data["sentence_tag"].append(sent_tag)
                if token.tag == "w":
                    data["token"].append(token.text.strip())
                    data["c5"].append(token.get("c5"))
                    data["lemma"].append(token.get("hw"))
                    data["pos"].append(token.get("pos"))
                elif token.tag == "c":
                    data["token"].append(token.text.strip())
                    data["c5"].append(token.get("c5"))
                    data["lemma"].append(None)
                    data["pos"].append("STOP")
                else:
                    data["token"].append(f"<{token.tag}/>")
                    data["c5"].append(None)
                    data["lemma"].append(None)
                    data["pos"].append(None)
            sent_tag = "I"

    # Neither block of metadata is appended in the loop above: the text columns
    # are one value each, and the speaker columns are cheaper to join on than to
    # look up and append per token.
    return (
        pl.DataFrame(data, schema=TOKEN_SCHEMA)
        .with_columns(
            pl.lit(value, dtype=TEXT_SCHEMA[column]).alias(column)
            for column, value in metadata.items()
        )
        .join(get_speakers(doc), on="speaker_id", how="left", maintain_order="left")
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

# Searching it back never loads the corpus, and the text and speaker metadata
# ride along on the results:
#
#     import polars as pl
#     import polars_corpus  # noqa: F401
#
#     corpus = pl.scan_parquet("bnc.parquet")
#     results = corpus.corpus.search("{take} * for granted")
#     print(results.concordance())
#
# A subcorpus is a filter over those same columns, so it costs a scan and no
# join -- here, fiction written by women:
#
#     fiction = corpus.filter(
#         pl.col("written_domain") == "Imaginative",
#         pl.col("author_sex") == "Female",
#     )
