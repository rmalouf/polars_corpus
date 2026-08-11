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


def matches(df: pl.DataFrame, query: str) -> list[tuple[int, int, str]]:
    """Search `df`, flattening the results to (start, end, matched text) tuples."""
    results = plc.search(df, query)
    return [
        (m.span.start, m.span.end, " ".join(df["token"][m.span.start : m.span.end]))
        for m in (results._matches if results is not None else ())
    ]


def bound(df: pl.DataFrame, match, var: str) -> str:
    """The text captured by variable `var` in `match`."""
    span = match.bindings[var]
    return " ".join(df["token"][span.start : span.end])


# Expected results shared by several tests, in corpus order.
ABLE_MATCHES = [
    (12, 13, "capable"),
    (36, 37, "able"),
    (38, 39, "table"),
    (40, 41, "capable"),
    (50, 51, "table"),
    (52, 53, "suitable"),
    (54, 55, "available"),
]

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

VBD_MATCHES = [
    (14, 15, "walked"),
    (25, 26, "parked"),
    (30, 31, "sang"),
    (61, 62, "came"),
]

NN_MATCHES = [
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
    (50, 51, "table"),
    (57, 58, "neighbour"),
    (59, 60, "neighbor"),
]

# The simplified SUBST class also covers the plural tag NNS.
NOUN_MATCHES = sorted(NN_MATCHES + [(46, 47, "mysteries")])


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
        ("*able", ABLE_MATCHES),
        # unlike *, + requires at least one character before "able"
        ("+able", [m for m in ABLE_MATCHES if m != (36, 37, "able")]),
        # --- bracketed alternatives ---
        ("[car,truck]", [(21, 22, "car"), (24, 25, "truck")]),
        ("[qu*,br*]", [(1, 2, "quick"), (2, 3, "brown")]),
        # a group is part of a word, not a whole token, and may be empty
        ("neighbo[u,]r", [(57, 58, "neighbour"), (59, 60, "neighbor")]),
        ("[s,][a,i]ng", [(29, 30, "sing"), (30, 31, "sang")]),
        (
            "??+[able,ability]",  # 3+ characters before the group, so not "table"
            [
                (12, 13, "capable"),
                (40, 41, "capable"),
                (52, 53, "suitable"),
                (54, 55, "available"),
            ],
        ),
        # whitespace around alternatives is layout, not part of the pattern
        ("[ car , truck ]", [(21, 22, "car"), (24, 25, "truck")]),
        (r"[\ truck,x]", []),  # an escaped space is a literal, so nothing matches
        # --- multi-word sequences ---
        ("quick brown", [(1, 3, "quick brown")]),
        ("quick br*", [(1, 3, "quick brown")]),
    ],
)
def test_word_patterns(sample_corpus, query, expected):
    assert matches(sample_corpus, query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("fox * over", [(3, 6, "fox jumps over")]),  # * is 0 or 1 token
        ("fox + over", [(3, 6, "fox jumps over")]),  # + is 1+ tokens
        ("The ++ fox", [(0, 4, "The quick brown fox")]),  # ++ is exactly 2 tokens
        ("A *** student", [(10, 14, "A very capable student")]),  # 0-3 tokens
        ("fox +++** dog", [(3, 9, "fox jumps over the lazy dog")]),  # 3-5 tokens
    ],
)
def test_gap_tokens(sample_corpus, query, expected):
    assert matches(sample_corpus, query) == expected


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
    assert matches(sample_corpus, query) == expected


def test_quantifier_vs_gap_whitespace_sensitivity(sample_corpus):
    """Whitespace disambiguates a group quantifier from a gap token.

    `(pattern)+` repeats the group; `(pattern) +` is the group followed by a
    mandatory filler token.
    """
    assert matches(sample_corpus, "(red)+ car") == [(20, 22, "red car")]
    assert matches(sample_corpus, "(red) + and") == [(20, 23, "red car and")]


@pytest.mark.parametrize(
    "query,expected",
    [
        # --- exact tags via _TAG ---
        ("fox_NN", [(3, 4, "fox")]),
        ("fox_nn", [(3, 4, "fox")]),  # tags match case-insensitively
        ("*ly_RB", [(15, 16, "slowly")]),  # wildcard in the word part
        ("sing_V*", [(29, 30, "sing")]),  # wildcard in the tag part
        ("walk[s,ed]_V*", [(14, 15, "walked")]),  # group in the word part
        ("sang_[VBD,VBN]", [(30, 31, "sang")]),  # group in the tag part
        ("_VBD", VBD_MATCHES),
        ("_NN", NN_MATCHES),
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
        # --- simplified classes via _{CLASS}, equivalent to a _VB* wildcard ---
        ("_{VERB}", VERB_MATCHES),
        ("_VB*", VERB_MATCHES),
        ("_{SUBST}", NOUN_MATCHES),
        ("walked_{VERB}", [(14, 15, "walked")]),
        ("*ly_{ADV}", [(15, 16, "slowly")]),
        ("the _{ADJ} dog", [(6, 9, "the lazy dog")]),
        # without braces a class name is a literal tag, which nothing here has
        ("_SUBST", []),
        ("fox_N", []),
        ("{box}_SUBST", []),
    ],
)
def test_pos_tag_patterns(sample_corpus, query, expected):
    assert matches(sample_corpus, query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("_ADJ", [(1, 2, "big")]),  # the tagset's own ADJ, not the class
        ("_{ADJ}", []),  # the class, which expands to AJ.*|JJ.*
        ("_PRON", [(4, 5, "it")]),
        ("_{PRON}", []),
    ],
)
def test_tagset_spelling_class_names(query, expected):
    """A tagset whose tags are spelled the way the simplified classes are"""
    ud_corpus = corpus(token="the big dog barks it", pos="DET ADJ NOUN VERB PRON")
    assert matches(ud_corpus, query) == expected


@pytest.mark.parametrize("query", ["_{FOO}", "walk_{FOO}", "{walk}_{FOO}"])
def test_unknown_pos_class_raises(query):
    with pytest.raises(ValueError, match="Unknown POS class 'FOO'"):
        simple_to_cqp(query)


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
    assert matches(sample_corpus, query) == expected


@pytest.mark.parametrize(
    "query,expected",
    [
        ("(car | truck)", [(21, 22, "car"), (24, 25, "truck")]),
        ("(red | blue) truck", [(23, 25, "blue truck")]),  # "red" precedes "car"
        ("(quick brown | red) fox", [(1, 4, "quick brown fox")]),
        ("(and)+ (schoolroom | mysteries)", [(44, 46, "and schoolroom")]),
        ("(*able | *ible)", ABLE_MATCHES),  # a branch that matches nothing
        # results come back in corpus order, not branch order
        (
            "(truck | fox | car | dog)",
            [(3, 4, "fox"), (8, 9, "dog"), (21, 22, "car"), (24, 25, "truck")],
        ),
        ("(_NN | _VBD)", sorted(NN_MATCHES + VBD_MATCHES)),
        # branches of unequal length: each jump must skip all the branches after it
        (
            "(red car | blue truck | dog)",
            [(8, 9, "dog"), (20, 22, "red car"), (23, 25, "blue truck")],
        ),
        ("(car | (truck | dog))", [(8, 9, "dog"), (21, 22, "car"), (24, 25, "truck")]),
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
    assert matches(sample_corpus, query) == expected


class TestEscapes:
    """Backslash-escaped metacharacters are literals, not wildcards"""

    @pytest.fixture
    def punctuation_corpus(self) -> pl.DataFrame:
        return corpus(
            token="x*x xyx x?x x+x a,b a/b New_York xx",
            lemma="x*x xyx x?x x+x a,b a/b new_york xx",
            pos="NN NN NN NN NN NN NNP NN",
        )

    @pytest.mark.parametrize(
        "query,expected",
        [
            (r"x\*x", [(0, 1, "x*x")]),
            (r"x\?x", [(2, 3, "x?x")]),
            (r"x\+x", [(3, 4, "x+x")]),
            # the same pattern unescaped: `*` is a wildcard again
            (
                "x*x",
                [
                    (0, 1, "x*x"),
                    (1, 2, "xyx"),
                    (2, 3, "x?x"),
                    (3, 4, "x+x"),
                    (7, 8, "xx"),
                ],
            ),
            (r"a\,b", [(4, 5, "a,b")]),
            (r"a\/b", [(5, 6, "a/b")]),
            (r"New\_York_NNP", [(6, 7, "New_York")]),
            # escapes inside alternative groups and lemma constraints
            (r"[x\*x,a\,b]", [(0, 1, "x*x"), (4, 5, "a,b")]),
            (r"{x\*x}", [(0, 1, "x*x")]),
            (r"{a\/b}", [(5, 6, "a/b")]),
            (r"{x\*x}_NN", [(0, 1, "x*x")]),
        ],
    )
    def test_escaped_metacharacters_match_literals(
        self, punctuation_corpus, query, expected
    ):
        assert matches(punctuation_corpus, query) == expected

    @pytest.mark.parametrize(
        "query,expected",
        [
            (r"x\*x", r'[token="x\*x"%c]'),
            (r"{x\*x}", r'[lemma="x\*x"%c]'),
            (r"{a\/b}", '[lemma="a/b"%c]'),
            ("{light/V}", '[lemma="light"%c & pos="V.*"%c]'),  # unescaped / separates
        ],
    )
    def test_translation_to_cqp(self, query, expected):
        assert simple_to_cqp(query) == expected


class TestBindings:
    """Variable bindings ($var: pattern) in simple queries"""

    @pytest.mark.parametrize(
        "query,var,expected",
        [
            ("$target: fox", "target", "fox"),
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
        "query,expected",
        [
            ("$x: fox", '$x: ([token="fox"%c])'),
            ("$a: quick $b: brown", '$a: ([token="quick"%c]) $b: ([token="brown"%c])'),
            ("$suffix: *able", '$suffix: ([token=".*able"%c])'),
            (
                "$phrase: (quick brown)",
                '$phrase: (([token="quick"%c] [token="brown"%c]))',
            ),
            ("($mods: very)+", '($mods: ([token="very"%c]))+'),
        ],
    )
    def test_translation_to_cqp(self, query, expected):
        assert simple_to_cqp(query) == expected
