import nltk.data
import polars as pl
import polars_corpus as plc
import pytest
from nltk.corpus import brown
from nltk.corpus.reader import (
    CorpusReader,
    PlaintextCorpusReader,
    TaggedCorpusReader,
    WordListCorpusReader,
)
from nltk.tokenize import LineTokenizer
from polars_corpus.convert import convert_bnc, from_nltk

# from_nltk emits a column only where the reader supplies one, so what a case
# has to vary is the reader's interface, not the text. Only the categorized
# case needs a downloaded corpus; the rest are reader classes over tmp_path,
# which keeps CI to one nltk package. LineTokenizer is there to keep the
# plaintext reader off punkt, whose data would be another download.
PLAIN = "the quick brown fox\njumps over the lazy dog\n"
TAGGED = "the/DT quick/JJ brown/JJ fox/NN\njumps/VBZ over/IN the/DT lazy/JJ dog/NN\n"
WORDS = "alpha\nbeta\ngamma\n"

NLTK_CORPORA = [
    pytest.param(
        lambda root: brown,
        ["token", "pos", "sentence_tag", "file_id", "category"],
        (1000, ("race", "NN", "I", "ca01", "news")),
        id="brown",
    ),
    pytest.param(
        lambda root: TaggedCorpusReader(root, r".*\.pos"),
        ["token", "pos", "sentence_tag", "file_id"],
        (0, ("the", "DT", "B", "a.pos")),
        id="tagged",
    ),
    pytest.param(
        lambda root: PlaintextCorpusReader(
            root, r".*\.txt", sent_tokenizer=LineTokenizer()
        ),
        ["token", "sentence_tag", "file_id"],
        (0, ("the", "B", "a.txt")),
        id="plaintext",
    ),
    pytest.param(
        lambda root: WordListCorpusReader(root, r".*\.words"),
        ["token", "file_id"],
        (0, ("alpha", "a.words")),
        id="wordlist",
    ),
]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A corpus directory holding one file per reader below.

    A reader may only read under a root on `nltk.data.path` or under a temp
    directory private to this user, which Linux's shared /tmp is not, so the
    directory has to be authorized rather than merely created.
    """
    monkeypatch.setattr(nltk.data, "path", [*nltk.data.path, str(tmp_path)])
    (tmp_path / "a.txt").write_text(PLAIN)
    (tmp_path / "a.pos").write_text(TAGGED)
    (tmp_path / "a.words").write_text(WORDS)
    return str(tmp_path)


@pytest.mark.parametrize("reader,columns,row", NLTK_CORPORA)
def test_from_nltk(reader, columns, row, root):
    reader = reader(root)
    c = from_nltk(reader)
    index, expected = row

    assert c.columns == columns
    assert c.get_column("file_id").n_unique() == len(reader.fileids())
    assert c.row(index) == expected

    if "sentence_tag" in columns:
        assert c.height == sum(len(sent) for sent in reader.sents())
        assert c.filter(pl.col("sentence_tag") == "B").height == len(reader.sents())
    else:
        assert c.height == len(reader.words())


def test_categories_partition_the_files():
    c = from_nltk(brown)
    n_sf_files = c.filter(pl.col("category") == "science_fiction")["file_id"].n_unique()
    assert n_sf_files == len(brown.fileids("science_fiction"))


def test_output_is_searchable_with_default_columns():
    """The tag column must be named `pos`, which is what search() looks for."""
    c = from_nltk(brown).head(2000)
    results = plc.search(c, "the _JJ _NN")
    assert results is not None and len(results.matches) > 0


def test_unsupported_reader(root):
    """A reader with no words() interface cannot be converted."""
    with pytest.raises(AttributeError):
        from_nltk(CorpusReader(root, r".*\.txt"))


# Two BNC texts cut down to the elements convert_bnc reads: the header's text
# id, creation date and classification codes, the speakers, and the tokens.
# A00 is written and KB0 spoken, so between them every column gets a value in
# one text and a null in the other.
WRITTEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<bncDoc xml:id="{file_id}">
 <teiHeader>
  <fileDesc><publicationStmt><idno type="bnc">{idno}</idno></publicationStmt></fileDesc>
  <profileDesc>
   <creation date="1985"/>
   <textClass>
    <catRef targets="WRI ALLTIM3 WRIAD1 WRIASE2 WRIATY3 WRIAUD3 WRIDOM1 WRIMED1 WRIPP1 WRISAM1 WRITAS0"/>
    <classCode scheme="DLEE">W fict prose</classCode>
   </textClass>
  </profileDesc>
 </teiHeader>
 <wtext type="FICTION">
  <div><p>
   <s n="1"><w c5="AT0" hw="the" pos="ART">The </w><w c5="AJ0" hw="quick" pos="ADJ">quick </w><w c5="NN1" hw="fox" pos="SUBST">fox</w><c c5="PUN">.</c></s>
   <s n="2"><w c5="PNP" hw="it" pos="PRON">It </w><w c5="VVD" hw="jump" pos="VERB">jumped</w><gap desc="picture"/><c c5="PUN">.</c></s>
  </p></div>
 </wtext>
</bncDoc>
"""

# The creation date is the "0000" of a text composed in no recorded year, and
# PS002 carries the "not recorded" code of every attribute that has one.
SPOKEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<bncDoc xml:id="KB0">
 <teiHeader>
  <fileDesc><publicationStmt><idno type="bnc">KB0</idno></publicationStmt></fileDesc>
  <profileDesc>
   <creation date="0000"/>
   <textClass>
    <catRef targets="SPO ALLTIM3 SDEAGE3 SDECLA2 SDESEX1 SPOLOG2 SPOREG1"/>
    <classCode scheme="DLEE">S conv</classCode>
   </textClass>
   <particDesc>
    <person xml:id="PS001" ageGroup="Ag3" sex="f" soc="C1" dialect="XLO" role="Wife"><persName>Ann</persName><occupation>teacher</occupation></person>
    <person xml:id="PS002" ageGroup="X" sex="u" soc="UU" dialect="NONE" role="unspecified"><persNote>a friend</persNote></person>
   </particDesc>
  </profileDesc>
 </teiHeader>
 <stext type="CONV">
  <div>
   <u who="PS001"><s n="1"><w c5="UH" hw="yes" pos="INTERJ">Yes </w><pause/><w c5="AV0" hw="well" pos="ADV">well</w><c c5="PUN">.</c></s></u>
   <u who="PS002"><s n="2"><w c5="PNP" hw="i" pos="PRON">I </w><unclear/><w c5="VVB" hw="know" pos="VERB">know</w></s></u>
  </div>
 </stext>
</bncDoc>
"""


@pytest.fixture(scope="module")
def bnc(tmp_path_factory):
    """The corpus the fixture texts convert to, read back from the Parquet file.

    Converting spawns worker processes, so it is done once for the module.
    """
    pytest.importorskip("lxml")
    pytest.importorskip("pyarrow")
    root = tmp_path_factory.mktemp("bnc_xml")
    for path, xml in [
        ("Texts/A/A0/A00.xml", WRITTEN_XML.format(file_id="A00", idno="A00")),
        ("Texts/K/KB/KB0.xml", SPOKEN_XML),
        # An earlier copy of A00 filed under a name of its own, the shape the
        # real corpus's G3C.xml takes: keeping it would give A00 two runs of
        # tokens with KB0's between them.
        ("Texts/G/G3/G3C.xml", WRITTEN_XML.format(file_id="G3C", idno="A00")),
    ]:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(xml)
    return convert_bnc(root, tmp_path_factory.mktemp("out") / "bnc.parquet").collect()


def test_convert_bnc_tokens(bnc):
    """One row per token, the texts in file id order and each in one run."""
    assert bnc.columns == list(plc.convert.BNC_SCHEMA)
    assert bnc.height == 15
    assert bnc.get_column("file_id").rle_id().max() == 1  # two texts, two runs
    assert bnc.row(0, named=True)["token"] == "The"
    assert bnc.get_column("sentence_tag").to_list().count("B") == 4
    # A gap, pause or unclear passage is a row named for its element.
    assert bnc.filter(pl.col("token") == "<gap/>").height == 1


@pytest.mark.parametrize(
    "column,written,spoken",
    [
        ("mode", "written", "spoken"),
        ("text_type", "FICTION", "CONV"),
        ("creation_year", 1985, None),
        # Decoded from the catRef codes, and null where the taxonomy does not
        # apply to the text.
        ("publication_date", "1985-1993", "1985-1993"),
        ("written_domain", "Imaginative", None),
        ("author_sex", "Female", None),
        ("region", None, "South"),
        ("respondent_social_class", None, "C1"),
    ],
)
def test_convert_bnc_metadata(bnc, column, written, spoken):
    """Each text carries one value of every metadata column on all its tokens."""
    for file_id, expected in [("A00", written), ("KB0", spoken)]:
        values = bnc.filter(pl.col("file_id") == file_id).get_column(column).unique()
        assert values.to_list() == [expected]


@pytest.mark.parametrize(
    "speaker_id,row",
    [
        (None, (None,) * 6),  # a written text has no speaker at all
        ("PS001", ("Female", "35-44", "C1", "London", "Wife", "Ann")),
        # PS002 is "not recorded" throughout, which reads back the same way.
        ("PS002", (None,) * 6),
    ],
)
def test_convert_bnc_speakers(bnc, speaker_id, row):
    """The speaker columns are joined on from <person>, with codes decoded."""
    columns = ["sex", "age_group", "social_class", "dialect", "role", "pers_name"]
    speaker = bnc.filter(pl.col("speaker_id").eq_missing(speaker_id))
    assert speaker.height > 0
    assert speaker.select(columns).unique().rows() == [row]


def test_convert_bnc_no_texts(tmp_path):
    """A directory with no `Texts` under it is an error."""
    pytest.importorskip("lxml")
    with pytest.raises(ValueError, match="No BNC texts found"):
        convert_bnc(tmp_path, tmp_path / "bnc.parquet")
