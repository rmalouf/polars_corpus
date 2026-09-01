"""Convert corpora in other formats into Polars frames and Parquet files."""

from __future__ import annotations

import importlib.util
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator, Optional, Union, cast

import polars as pl

if TYPE_CHECKING:
    import pyarrow.parquet as pq
    from nltk.corpus.reader.api import CategorizedCorpusReader, CorpusReader

__all__ = ["from_nltk", "convert_bnc"]

PathType = Union[str, PathLike[str]]


def from_nltk(corpus: CorpusReader) -> pl.DataFrame:
    """
    Read an NLTK corpus into a Polars DataFrame, one row per token.

    Works with any corpus NLTK can read: Brown, the Gutenberg texts, anything
    under `nltk.corpus`, or a reader pointed at a directory of your own.

    The columns depend on what the reader offers. A tagged corpus gets a `pos`
    column, a corpus read as sentences gets a `sentence_tag` column, and a
    categorized corpus gets a `category` column.

    Parameters
    ----------
    corpus : CorpusReader
        Any NLTK corpus reader, e.g. `nltk.corpus.brown` or a
        `PlaintextCorpusReader` over a directory of your own.

    Returns
    -------
    pl.DataFrame
        One row per token, in corpus order, with as many of these columns as
        the reader can supply:

        - `token` : the word itself
        - `pos` : its part-of-speech tag, if the corpus is tagged
        - `sentence_tag` : "B" on the first token of each sentence and "I" on
          the rest, if the reader reads sentences
        - `file_id` : the file the token came from
        - `category` : the file's category, if the corpus is categorized

    Raises
    ------
    AttributeError
        If the reader exposes none of `sents()`, `tagged_words()` or
        `words()`, so its tokens cannot be read.

    Examples
    --------
    >>> import nltk
    >>> import polars_corpus as plc
    >>> brown = plc.from_nltk(nltk.corpus.brown)
    >>> # `category` is there because Brown is categorized:
    >>> brown.group_by("category").len()
    """
    category_dict = dict()
    if hasattr(corpus, "categories"):
        # CategorizedCorpusReader is a mixin, so a categorized corpus is only
        # identifiable by duck-typing. It supplies categories() and widens
        # fileids(), which on the plain reader takes no arguments.
        categorized = cast("CategorizedCorpusReader", corpus)
        for category in categorized.categories():
            for file_id in categorized.fileids(category):
                category_dict[file_id] = category
    corpus_data = []
    for file_id in corpus.fileids():
        for token_dict in _convert_file(corpus, file_id):
            token_dict["file_id"] = file_id
            if file_id in category_dict:
                token_dict["category"] = category_dict[file_id]
            corpus_data.append(token_dict)
    return pl.DataFrame(corpus_data)


def _convert_file(corpus: CorpusReader, file_id: str) -> Generator[dict[str, str]]:
    if hasattr(corpus, "sents"):
        if hasattr(corpus, "tagged_sents"):
            sentences = corpus.tagged_sents(file_id)
        else:
            sentences = corpus.sents(file_id)
        for sentence in sentences:
            first_word = True
            for token_dict in _convert_token(sentence):
                if first_word:
                    token_dict["sentence_tag"] = "B"
                    first_word = False
                else:
                    token_dict["sentence_tag"] = "I"
                yield token_dict
    else:
        if hasattr(corpus, "tagged_words"):
            tokens = corpus.tagged_words(file_id)
        elif hasattr(corpus, "words"):
            tokens = corpus.words(file_id)
        else:
            raise AttributeError(
                f"{type(corpus).__name__} exposes none of sents(), tagged_words(), "
                "or words(), so its tokens cannot be read"
            )
        for token_dict in _convert_token(tokens):
            yield token_dict


def _convert_token(tokens: list[Any]) -> Generator[dict[str, str]]:
    for token in tokens:
        token_dict = {}
        if type(token) is tuple:
            token_dict["token"] = token[0]
            token_dict["pos"] = token[1]
        else:
            token_dict["token"] = token
        yield token_dict


# The BNC XML edition is at
# https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554. The
# column names and labels below are the Reference Guide's, from its tables in
# section 1 (Design of the corpus), so a column here is searchable in the Guide
# by its own name: http://www.natcorp.ox.ac.uk/docs/URG/index.html
#
# The file is laid out for the lazy search path in `matcher.py`, which asks for
#
#   * a `file_id` column, since that is what it chunks the corpus by;
#   * every token of a text in one contiguous run, so texts go in whole, one at
#     a time, in sorted order;
#   * row groups aligned to text boundaries and small enough that slicing to a
#     handful of texts decodes little around them.

# Parsing runs in worker processes: lxml frees the GIL, but that is only a
# fifth of the work here, and threads measure no faster than serial.
N_WORKERS = 8

# Tokens to aim for per row group, the unit a scan prunes to and so the
# granularity a lazy `slice` can seek at. Smaller groups decode less around a
# concordance, at the cost of more metadata and worse compression.
ROW_GROUP_TOKENS = 250_000

# The <person> attributes worth keeping, as column -> (attribute, the label for
# each code). The code for "not recorded" maps to None, so a null test covers
# both an unrecorded speaker and a written text with no speaker at all; a code
# no table names is kept as it stands, which is how `social_class` and `role`
# pass their values through. All five attributes are on all 6124 <person>
# elements.
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
    # demographically sampled text. 79 values, all of them already words, so
    # only the two that mean "not recorded" are named.
    "role": ("role", {"unspecified": None, "?": None}),
}

# The rest of the speaker metadata is in child elements. `age` and `educ` are
# left out: `educ` is "X" for all but a handful of speakers, and `age` is free
# text ("40", "30+") that `age_group` already bins.
SPEAKER_ELEMENTS = {
    "pers_name": "persName",
    "occupation": "occupation",
    "pers_note": "persNote",
}

# Each text carries one code from each of the classification taxonomies that
# apply to it, all run together in the `targets` attribute of its <catRef>.
# This maps column -> (taxonomy prefix, the label for each code), with the
# BNC's code for "not recorded" mapped to None: a null then covers both an
# unclassified text and a taxonomy that does not apply.
#
# Three taxonomies are left out. WRILEV (perceived difficulty) and WRISTA
# (estimated circulation) "were incorrectly differentiated during the
# preparation of the corpus and cannot be relied on" -- the Guide's own words.
# ALLTYP repeats what `mode`, `text_type` and `written_medium` say.
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
# repetitive columns on disk, and Categorical costs memory once the corpus is
# read back.
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

# One value per text, so these are set as literals. The two dates are separate
# facts: `creation_year` is <creation date>, "the year of original
# composition", and `publication_date` is the band the text was classified
# under, from the date of publication for a written text and of the recording
# for a spoken one. 383 texts are banded but composed in no recorded year, and
# 2 the other way round.
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
# is what completes BNC_SCHEMA.
SPEAKER_SCHEMA = pl.Schema(
    {"speaker_id": pl.String}
    | {column: pl.String for column in SPEAKER_ATTRS | SPEAKER_ELEMENTS}
)

BNC_SCHEMA = pl.Schema(
    TOKEN_SCHEMA
    | TEXT_SCHEMA
    | {c: t for c, t in SPEAKER_SCHEMA.items() if c != "speaker_id"}
)


def _speakers(doc: Any) -> pl.DataFrame:
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


def _text_class(doc: Any) -> dict[str, Optional[str]]:
    """The document's classification codes, decoded to labels."""
    codes = doc.xpath("//catRef/@targets")[0].split()
    found = {}
    for column, (prefix, labels) in CATEGORIES.items():
        # A handful of texts are missing a code the rest of their sample has,
        # which reads back the same as one recorded as unknown.
        code = next((c for c in codes if c.startswith(prefix)), None)
        found[column] = labels[int(code.removeprefix(prefix))] if code else None
    return found


def _parse_text(path: Path) -> Optional[pl.DataFrame]:
    """Parse one BNC XML file into a frame of tokens carrying its metadata.

    Runs in a worker process, so it imports lxml itself. `None` means the file
    is a stray copy of a text filed elsewhere, to be skipped.
    """
    from lxml import etree

    doc = etree.parse(str(path))
    docid = doc.xpath('//idno[@type="bnc"]')[0].text
    # G3C.xml is an earlier copy of HWX.xml, header and all, so two files claim
    # the id HWX, which would give that id two runs of tokens far apart. The
    # stray copy is the one filed under a name other than the id it carries, so
    # dropping it here keeps HWX.xml, the corrected December 2006 text.
    if docid != path.stem:
        return None
    if text := doc.xpath("//wtext"):
        text_mode = "written"
    elif text := doc.xpath("//stext"):
        text_mode = "spoken"
    else:
        raise ValueError(f"{path} holds neither a <wtext> nor an <stext>")

    # "0000" is what the header gives for a text whose date is unknown.
    created = doc.xpath("//creation/@date")[0]
    metadata = {
        "file_id": docid,
        "mode": text_mode,
        "text_type": text[0].get("type"),
        "genre": doc.xpath("//classCode")[0].text,
        "creation_year": None if created == "0000" else int(created),
    } | _text_class(doc)

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

    # The text columns are one value each, and the speaker columns are cheaper
    # to join on than to look up and append per token.
    return (
        pl.DataFrame(data, schema=TOKEN_SCHEMA)
        .with_columns(
            pl.lit(value, dtype=TEXT_SCHEMA[column]).alias(column)
            for column, value in metadata.items()
        )
        .join(_speakers(doc), on="speaker_id", how="left", maintain_order="left")
    )


def _parse_texts(paths: list[Path]) -> Generator[pl.DataFrame]:
    """Parse `paths` in worker processes, yielding the frames in path order.

    Order is what keeps each file id in a single run. At most `2 * N_WORKERS`
    texts are in flight, so the writer sets the pace and memory stays flat.
    """
    remaining = iter(paths)
    with ProcessPoolExecutor(min(N_WORKERS, len(paths))) as pool:
        pending = deque(
            pool.submit(_parse_text, path) for path in islice(remaining, 2 * N_WORKERS)
        )
        while pending:
            df = pending.popleft().result()
            for path in islice(remaining, 1):
                pending.append(pool.submit(_parse_text, path))
            if df is not None:  # a stray copy of another text; see _parse_text
                yield df


def _write_row_group(writer: pq.ParquetWriter, batch: list[pl.DataFrame]) -> None:
    """Write a batch of whole texts as one row group."""
    table = pl.concat(batch).to_arrow()
    # The explicit size puts the seam between two texts; pyarrow's default
    # caps a row group at 1M rows.
    writer.write_table(table, row_group_size=table.num_rows)


def convert_bnc(bnc_root: PathType, output_path: PathType) -> pl.LazyFrame:
    """
    Convert the XML edition of the British National Corpus into a Parquet file.

    The corpus is read one text at a time and appended to the Parquet file as
    it goes, so converting it holds a handful of texts at once. Text and
    speaker metadata are written onto every token, so restricting a search to
    a subcorpus is a filter.

    Parameters
    ----------
    bnc_root : str or Path
        The root of the BNC XML distribution, the directory holding `Texts`.
    output_path : str or Path
        Parquet file to write. An existing file is overwritten.

    Returns
    -------
    LazyFrame
        A scan of the file just written, one row per token, the texts in file
        id order and the tokens of a text in the order they appear in it, with
        the columns

        - `token`, `lemma`, `c5` : the word, its headword, and its CLAWS5 tag
        - `pos` : the simplified tag, or "STOP" on a punctuation mark
        - `sentence_tag` : "B" on the first token of a sentence, "I" on the rest
        - `file_id` : the three-character text id, e.g. "A00"
        - `mode`, `text_type`, `genre` : what the text is
        - `creation_year` : the year it was composed, null where the header
          records none
        - one column per classification taxonomy, e.g. `written_domain`,
          `author_sex`, `region`, holding the label the text's code stands for
          and null where the taxonomy does not apply to it
        - `speaker_id`, `sex`, `age_group`, `social_class`, `dialect`, `role`,
          `pers_name`, `occupation`, `pers_note` : who spoke the token, null
          throughout a written text

    Raises
    ------
    ImportError
        If lxml or pyarrow is not installed. Both are the `examples` extra.
    ValueError
        If `bnc_root` holds no `Texts` directory with XML files under it, or a
        file there holds neither a `<wtext>` nor an `<stext>` element.

    Notes
    -----
    A `<gap>`, `<unclear>`, `<pause>`, `<vocal>` or `<event>` element is one
    row whose `token` names it, e.g. `<pause/>`, and whose `lemma`, `pos` and
    `c5` are null.

    Texts are parsed in worker processes, so a script that calls this guards
    the call with `if __name__ == "__main__":`.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> corpus = plc.convert_bnc("/Volumes/Corpora/bnc_xml", "bnc.parquet")
    >>> # Later runs scan the file it wrote:
    >>> corpus = pl.scan_parquet("bnc.parquet")
    >>> corpus.corpus.search("{take} * for granted").concordance()
    >>> # A subcorpus is a filter over the metadata columns:
    >>> fiction = corpus.filter(
    ...     pl.col("written_domain") == "Imaginative",
    ...     pl.col("author_sex") == "Female",
    ... )
    """
    # The workers import lxml themselves, so it is checked for here.
    for package in ("lxml", "pyarrow"):
        if importlib.util.find_spec(package) is None:
            raise ImportError(
                f"convert_bnc needs {package}, which is not installed: "
                "pip install polars-corpus[examples]"
            )
    import pyarrow.parquet as pq

    texts = Path(bnc_root) / "Texts"
    paths = sorted(texts.glob("**/*.xml"))
    if not paths:
        raise ValueError(f"No BNC texts found under {texts}")
    parquet = Path(output_path)

    schema = pl.DataFrame(schema=BNC_SCHEMA).to_arrow().schema
    batch: list[pl.DataFrame] = []
    batch_tokens = 0
    with pq.ParquetWriter(parquet, schema, compression="zstd") as writer:
        for df in _parse_texts(paths):
            # Close the group before the text that would overrun it, so only
            # a text longer than the target lands in one alone.
            if batch and batch_tokens + df.height > ROW_GROUP_TOKENS:
                _write_row_group(writer, batch)
                batch, batch_tokens = [], 0
            batch.append(df)
            batch_tokens += df.height
        if batch:
            _write_row_group(writer, batch)
    return pl.scan_parquet(parquet)
