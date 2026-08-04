import polars as pl
import polars_corpus as plc
import pytest
from polars_corpus.simple_parser import simple_to_cqp

from .helpers import corpus

TOKENS = (
    "The quick brown fox jumps over the lazy dog . "
    "A very capable student walked slowly to school . "
    "The red car and blue truck parked outside . "
    "I sing sang song yesterday . "
    "They are able to table the capable motion . "
    "Voodoo and schoolroom mysteries . "
    "The big table is suitable and available . "
    "My neighbour and neighbor both came ."
)
POS = (
    "DT JJ JJ NN VBZ IN DT JJ NN . DT RB JJ NN VBD RB TO NN . "
    "DT JJ NN CC JJ NN VBD RB . PRP VBP VBD NN RB . "
    "PRP VBP JJ TO VB DT JJ NN . NN CC NN NNS . "
    "DT JJ NN VBZ JJ CC JJ . PRP$ NN CC NN DT VBD ."
)
LEMMA = (
    "the quick brown fox jump over the lazy dog . "
    "a very capable student walk slowly to school . "
    "the red car and blue truck park outside . "
    "i sing sing song yesterday . "
    "they be able to table the capable motion . "
    "voodoo and schoolroom mystery . "
    "the big table be suitable and available . "
    "my neighbour and neighbor both come ."
)


@pytest.fixture
def sample_corpus() -> pl.DataFrame:
    return corpus(token=TOKENS, pos=POS, lemma=LEMMA)


def found(df: pl.DataFrame, results) -> list[tuple[int, int, str]]:
    """Flatten search results to (start, end, matched text) tuples."""
    if results is None:
        return []
    return [
        (m.span.start, m.span.end, " ".join(df["token"][m.span.start : m.span.end]))
        for m in results._matches
    ]


def bound(df: pl.DataFrame, match, var: str) -> str:
    """The text captured by variable `var` in `match`."""
    span = match.bindings[var]
    return " ".join(df["token"][span.start : span.end])


@pytest.mark.parametrize(
    "query,expected",
    [
        # --- literal word forms (case-insensitive by default) ---
        ("fox", [(3, 4, "fox")]),
        (
            "the",
            [
                (0, 1, "The"),
                (6, 7, "the"),
                (19, 20, "The"),
                (39, 40, "the"),
                (48, 49, "The"),
            ],
        ),
        # --- wildcards: ? is one char, * is zero or more, + is one or more ---
        ("fo?", [(3, 4, "fox")]),
        ("*ick", [(1, 2, "quick")]),
        ("qu*", [(1, 2, "quick")]),
        ("+uck", [(24, 25, "truck")]),
        ("s?ng", [(29, 30, "sing"), (30, 31, "sang"), (31, 32, "song")]),
        (
            "*able",
            [
                (12, 13, "capable"),
                (36, 37, "able"),
                (38, 39, "table"),
                (40, 41, "capable"),
                (50, 51, "table"),
                (52, 53, "suitable"),
                (54, 55, "available"),
            ],
        ),
        (
            "+able",  # unlike *, requires at least one character before "able"
            [
                (12, 13, "capable"),
                (38, 39, "table"),
                (40, 41, "capable"),
                (50, 51, "table"),
                (52, 53, "suitable"),
                (54, 55, "available"),
            ],
        ),
        # --- bracketed alternatives ---
        ("[car,truck]", [(21, 22, "car"), (24, 25, "truck")]),
        ("[qu*,br*]", [(1, 2, "quick"), (2, 3, "brown")]),
        ("[neighbour,neighbor]", [(57, 58, "neighbour"), (59, 60, "neighbor")]),
        # --- multi-word sequences ---
        ("quick brown", [(1, 3, "quick brown")]),
        ("the lazy dog", [(6, 9, "the lazy dog")]),
        ("quick br*", [(1, 3, "quick brown")]),
    ],
)
def test_word_patterns(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("fox * over", [(3, 6, "fox jumps over")]),  # * is 0 or 1 token
        ("fox + over", [(3, 6, "fox jumps over")]),  # + is 1+ tokens
        ("red * and", [(20, 23, "red car and")]),
        ("The ++ fox", [(0, 4, "The quick brown fox")]),  # ++ is exactly 2 tokens
        ("A *** student", [(10, 14, "A very capable student")]),  # 0-3 tokens
        ("fox +++** dog", [(3, 9, "fox jumps over the lazy dog")]),  # 3-5 tokens
    ],
)
def test_gap_tokens(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("(very)? capable", [(11, 13, "very capable"), (40, 41, "capable")]),
        ("the (lazy)+", [(6, 8, "the lazy")]),
        ("The (quick)* brown", [(0, 3, "The quick brown")]),
        ("The (quick){1} brown", [(0, 3, "The quick brown")]),
        ("The (quick){1,2} brown", [(0, 3, "The quick brown")]),
        ("(quick brown)? fox", [(1, 4, "quick brown fox")]),
        ("(fox * over)?", [(3, 6, "fox jumps over")]),
    ],
)
def test_group_quantifiers(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


def test_quantifier_vs_gap_whitespace_sensitivity(sample_corpus):
    """Whitespace disambiguates a group quantifier from a gap token.

    `(pattern)+` repeats the group; `(pattern) +` is the group followed by a
    mandatory filler token.
    """
    quantifier = plc.search(sample_corpus, "(red)+ car")
    assert found(sample_corpus, quantifier) == [(20, 22, "red car")]

    gap = plc.search(sample_corpus, "(red) + and")
    assert found(sample_corpus, gap) == [(20, 23, "red car and")]


VERB_MATCHES = [
    (4, 5, "jumps"),  # VBZ
    (14, 15, "walked"),  # VBD
    (25, 26, "parked"),  # VBD
    (29, 30, "sing"),  # VBP
    (30, 31, "sang"),  # VBD
    (35, 36, "are"),  # VBP
    (38, 39, "table"),  # VB
    (51, 52, "is"),  # VBZ
    (61, 62, "came"),  # VBD
]

NOUN_MATCHES = [
    (3, 4, "fox"),
    (8, 9, "dog"),
    (13, 14, "student"),
    (17, 18, "school"),
    (21, 22, "car"),
    (24, 25, "truck"),
    (31, 32, "song"),
    (41, 42, "motion"),
    (43, 44, "Voodoo"),
    (45, 46, "schoolroom"),
    (46, 47, "mysteries"),  # NNS
    (50, 51, "table"),
    (57, 58, "neighbour"),
    (59, 60, "neighbor"),
]


@pytest.mark.parametrize(
    "query,expected",
    [
        # --- exact tags via _TAG ---
        ("fox_NN", [(3, 4, "fox")]),
        ("*ly_RB", [(15, 16, "slowly")]),  # wildcard in the word part
        ("sing_V*", [(29, 30, "sing")]),  # wildcard in the tag part
        (
            "_VBD",
            [
                (14, 15, "walked"),
                (25, 26, "parked"),
                (30, 31, "sang"),
                (61, 62, "came"),
            ],
        ),
        ("_NN", [m for m in NOUN_MATCHES if m != (46, 47, "mysteries")]),
        ("the _JJ dog", [(6, 9, "the lazy dog")]),
        (
            "_DT _JJ _NN",
            [
                (6, 9, "the lazy dog"),
                (19, 22, "The red car"),
                (39, 42, "the capable motion"),
                (48, 51, "The big table"),
            ],
        ),
        # --- simplified tags via _{TAG}, equivalent to the _VB* style wildcard ---
        ("_{VERB}", VERB_MATCHES),
        ("_VB*", VERB_MATCHES),
        ("_{SUBST}", NOUN_MATCHES),
        ("walked_{VERB}", [(14, 15, "walked")]),
        ("*ly_{ADV}", [(15, 16, "slowly")]),
        ("the _{ADJ} dog", [(6, 9, "the lazy dog")]),
    ],
)
def test_pos_tag_patterns(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        # "sing" and "sang" share the lemma "sing"
        ("{sing}", [(29, 30, "sing"), (30, 31, "sang")]),
        ("{walk}", [(14, 15, "walked")]),
        ("{capable/A}", [(12, 13, "capable"), (40, 41, "capable")]),
        ("{table/N}", [(50, 51, "table")]),  # the noun, not the VB at index 38
        ("{be/V}", [(35, 36, "are"), (51, 52, "is")]),
        # lemma combined with sequence and gap operators
        ("{sing} sang", [(29, 31, "sing sang")]),
        ("{be} {able}", [(35, 37, "are able")]),
        ("{be} * suitable", [(51, 53, "is suitable")]),
        # lemma plus a tag constraint, in each of the three tag spellings
        ("{sing}_VBD", [(30, 31, "sang")]),  # excludes "sing", which is VBP
        ("{be}_V*", [(35, 36, "are"), (51, 52, "is")]),
        ("{be}_{VERB}", [(35, 36, "are"), (51, 52, "is")]),
        ("{mystery}_{SUBST}", [(46, 47, "mysteries")]),
    ],
)
def test_lemma_patterns(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("(car | truck)", [(21, 22, "car"), (24, 25, "truck")]),
        ("(red | blue) truck", [(23, 25, "blue truck")]),  # "red" precedes "car"
        ("(quick brown | red) fox", [(1, 4, "quick brown fox")]),
        ("(and)+ (schoolroom | mysteries)", [(44, 46, "and schoolroom")]),
        (
            "(*able | *ible)",
            [
                (12, 13, "capable"),
                (36, 37, "able"),
                (38, 39, "table"),
                (40, 41, "capable"),
                (50, 51, "table"),
                (52, 53, "suitable"),
                (54, 55, "available"),
            ],
        ),
        (
            "the (_{ADJ} | _{SUBST})",
            [
                (0, 2, "The quick"),
                (6, 8, "the lazy"),
                (19, 21, "The red"),
                (39, 41, "the capable"),
                (48, 50, "The big"),
            ],
        ),
    ],
)
def test_disjunction(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


def test_disjunction_of_pos_tags(sample_corpus):
    """(_NN | _VBD) is the union of the two tag sets, in corpus order"""
    expected = sorted(
        [m for m in NOUN_MATCHES if m != (46, 47, "mysteries")]
        + [(14, 15, "walked"), (25, 26, "parked"), (30, 31, "sang"), (61, 62, "came")]
    )
    assert found(sample_corpus, plc.search(sample_corpus, "(_NN | _VBD)")) == expected


@pytest.mark.xfail(
    reason="alternation with more than two branches drops the middle branches; "
    "(a | b | c) matches only a and c",
    strict=True,
)
@pytest.mark.parametrize(
    "query,expected",
    [
        (
            "(car | truck | dog)",
            [(8, 9, "dog"), (21, 22, "car"), (24, 25, "truck")],
        ),
        (
            "(fox | dog | car | truck)",
            [(3, 4, "fox"), (8, 9, "dog"), (21, 22, "car"), (24, 25, "truck")],
        ),
    ],
)
def test_n_way_disjunction(sample_corpus, query, expected):
    assert found(sample_corpus, plc.search(sample_corpus, query)) == expected


class TestBindings:
    """Variable bindings ($var: pattern) in simple queries"""

    @pytest.mark.parametrize(
        "query,var,expected",
        [
            ("$target: fox", "target", "fox"),
            ("$word: quick", "word", "quick"),
            ("$suffix: *able", "suffix", "capable"),
            ("$pos: _NN", "pos", "fox"),
            ("$lemma: {sing}", "lemma", "sing"),
            ("$tagged: walked_VBD", "tagged", "walked"),
            ("$phrase: (quick brown) fox", "phrase", "quick brown"),
            ("($mods: very)+ capable", "mods", "very"),
            ("$vehicle: [car,truck]", "vehicle", "car"),
        ],
    )
    def test_binding_captures_expected_text(self, sample_corpus, query, var, expected):
        results = plc.search(sample_corpus, query)
        assert results is not None
        assert bound(sample_corpus, results._matches[0], var) == expected

    def test_multiple_variables(self, sample_corpus):
        results = plc.search(sample_corpus, "$color: brown $noun: fox")
        match = results._matches[0]
        assert bound(sample_corpus, match, "color") == "brown"
        assert bound(sample_corpus, match, "noun") == "fox"

    @pytest.mark.parametrize(
        "query,expected_fragments",
        [
            ("$x: fox", ["$x: ([token=", "fox"]),
            ("$a: quick $b: brown", ["$a:", "$b:"]),
            ("$suffix: *able", ["$suffix:", ".*able"]),
            ("$phrase: (quick brown)", ["$phrase: ("]),
        ],
    )
    def test_translation_to_cqp(self, query, expected_fragments):
        cqp = simple_to_cqp(query)
        for fragment in expected_fragments:
            assert fragment in cqp
