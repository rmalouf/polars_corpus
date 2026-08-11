"""Parser for BNCweb-style Simple Query Language.

This module implements a parser for the simple query syntax used in BNCweb,
which provides an alternative to CQP syntax for corpus searches. The parser
translates simple queries directly to CQP expressions.

Supports variable bindings using $varname: pattern syntax, which translates
to CQP's $varname: (pattern) format with automatic parenthesis wrapping.

`_GRAMMAR` below is the grammar; docs/simple_query.md documents the language it
accepts.
"""

from __future__ import annotations

import re
from typing import Any

from lark import Lark, Transformer
from lark.exceptions import VisitError

from .utils import check_choice

__all__ = ["simple_to_cqp"]


# Simplified POS classes, written `_{CLASS}` -- supports both BNC CLAWS-5 and
# Penn Treebank tagsets. An unbraced `_TAG` is always a literal tag pattern, so
# that a corpus whose tagset uses these names (`ADJ`, `PRON`, ... in Universal
# Dependencies) can still be searched for them.
_POS_MAPPING = {
    "V": "V.*",
    "VERB": "V.*",
    "N": "N.*",
    "SUBST": "N.*",
    "A": "(AJ.*|JJ.*)",
    "ADJ": "(AJ.*|JJ.*)",
    "ADV": "(AV.*|RB.*)",
    "ART": "(AT.*|DT)",
    "CONJ": "(CJ.*|CC)",
    "PREP": "(PR.*|IN|TO)",
    "PRON": "(PN.*|PRP.*)",
    "INT": "(ITJ|UH)",
    "INTERJ": "(ITJ|UH)",
    "STOP": "PU.*",
    "UNC": "UNC",
}
_POS_CLASSES = sorted(name.lower() for name in _POS_MAPPING)


# Regex fragments (injected into the Lark grammar). `/` must be escaped as `\/`
# because Lark uses `/` as the regex-literal delimiter in its grammar syntax.
# Characters that carry syntax; a word can hold any other character, punctuation
# and non-ASCII letters included, and these too once a backslash escapes them.
_SPECIAL = r"\s?*+,:$\/()|\[\]{}_\\<>"
_ESC = r"\\[^A-Za-z0-9]"  # a \X escape sequence: X is any non-alphanumeric
_WC = rf"[^{_SPECIAL}]"  # word character (non-wildcard)
_WL = r"[?*+]"  # wildcard
_ALT = r"\[[^\]]*\]"  # bracketed alternative group, e.g. `[u,]`
_NW = f"(?:{_ESC}|{_WC}|{_ALT})"  # non-wildcard part (escape, plain char, or group)
_PC = f"(?:{_ESC}|{_WC}|{_WL}|{_ALT})"  # part char (including wildcards)
_LI = rf"(?:{_ESC}|[^\\}}])+"  # inside `{...}`: anything but a bare backslash or brace
_TAG = rf"(?:\{{[A-Za-z]+\}}|{_PC}+)"  # a `_TAG` suffix: `{{CLASS}}` or a tag pattern


_GRAMMAR = rf"""
start: seq
seq: item+
?item: binding | group | atom
binding: BINDING_HEAD (group | atom)
group: _LPAREN seq (_PIPE seq)* RPAREN_QUANT
?atom: LEMMA_POS_TAG | LEMMA | POS_TAG | GAPS | WORD

BINDING_HEAD: /\$[A-Za-z][A-Za-z0-9_]*:/
RPAREN_QUANT: /\)(?:[?+*]|\{{\d+(?:,\d+)?\}})?/
_LPAREN: "("
_PIPE: "|"

LEMMA_POS_TAG: /\{{{_LI}\}}_{_TAG}/
LEMMA: /\{{{_LI}\}}/
// POS_TAG and WORD overlap (e.g. "fox_NN" could start with a WORD match on
// "fox"); give POS_TAG higher priority so Lark's lexer prefers it over the
// shorter WORD match. Longest-match alone isn't reliable here with complex
// character classes in the Lark basic lexer.
POS_TAG.2: /{_PC}*_{_TAG}/
GAPS: /[+*]+/
WORD: /{_PC}*{_NW}{_PC}*/

%ignore /[ \t\r\n]+/
"""


# A terminal reaches the transformer as one undivided string, so each is matched
# again here to recover its parts. Built from the fragments the grammar itself
# uses, so the two cannot drift; the greedy `_LI` is what stops an escaped brace
# inside a lemma from being read as the closing one.
_POS_TAG_PARTS = re.compile(rf"(?P<word>{_PC}*)_(?P<tag>{_TAG})", re.S)
_LEMMA_PARTS = re.compile(rf"\{{(?P<lemma>{_LI})\}}(?:_(?P<tag>{_TAG}))?", re.S)

# Splits a pattern into escapes, alternative groups, and single characters.
_PARTS = re.compile(rf"{_ESC}|{_ALT}|.", re.S)
# Whitespace padding an alternative, which is layout rather than part of the
# pattern. An escaped space (`\ `) is a literal and so is left alone.
_ALT_PAD = re.compile(r"^\s+|(?<!\\)\s+$")
# Separators that only separate when unescaped: `,` between alternatives,
# `/` between a lemma and its POS class.
_ALT_SEP = re.compile(r"(?<!\\),")
_LEMMA_SEP = re.compile(r"(?<!\\)/")

_WILDCARDS = {"?": ".", "*": ".*", "+": ".+"}


def _literal(char: str) -> str:
    """Regex matching `char` itself, safe to place in a CQP double-quoted value."""
    return r"\"" if char == '"' else re.escape(char)


def _to_regex(pattern: str) -> str:
    r"""Convert a simple query pattern to regex.

    Wildcards are `?` (one character), `*` (zero or more) and `+` (one or more);
    `[a,b]` alternates between its comma-separated parts, which may be empty or
    hold wildcards of their own; and a backslash makes the next character a
    literal, so `x\*x` matches `x*x`.
    """
    parts = []
    for part in _PARTS.findall(pattern):
        if part.startswith("["):  # `[a,b]` group -- alternatives are patterns too
            alts = "|".join(
                _to_regex(_ALT_PAD.sub("", alt)) for alt in _ALT_SEP.split(part[1:-1])
            )
            parts.append(f"(?:{alts})")
        elif len(part) == 2:  # `\X` escape: the escaped character is a literal
            parts.append(_literal(part[1]))
        else:
            parts.append(_WILDCARDS.get(part, _literal(part)))
    return "".join(parts)


def _split_lemma(inner: str) -> tuple[str, str]:
    """Split `lemma[/CLASS]` on the first unescaped slash."""
    parts = _LEMMA_SEP.split(inner, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _gap_tokens(gap_str: str) -> str:
    """Expand a run of gap markers: `+` requires a token, `*` allows one."""
    low = gap_str.count("+")
    high = low + gap_str.count("*")
    if low == high:
        return f"[]{{{low}}}"
    if low == 0 and high == 1:
        return "[]?"
    return f"[]{{{low},{high}}}"


def _resolve_pos_tag(raw: str) -> str:
    """Convert a `_TAG` suffix to a CQP pattern.

    `{CLASS}` names a simplified class; anything else is a literal tag pattern,
    even where it spells one of the class names.
    """
    if raw.startswith("{") and raw.endswith("}"):
        return _pos_class(raw[1:-1])
    return _to_regex(raw)


def _pos_class(name: str) -> str:
    """Expand a simplified POS class name, e.g. `SUBST` -> `N.*`."""
    return _POS_MAPPING[check_choice(name, _POS_CLASSES, param="POS class").upper()]


class SimpleCompiler(Transformer):
    def __init__(self, token_column: str, pos_column: str, lemma_column: str) -> None:
        super().__init__()
        self.token_column = token_column
        self.pos_column = pos_column
        self.lemma_column = lemma_column

    def _token(self, word: str = "", lemma: str = "", pos: str = "") -> str:
        """Join a terminal's translated patterns into one CQP token constraint.

        Whichever parts the terminal supplied; all matching is case-insensitive.
        """
        columns = (
            (self.token_column, word),
            (self.lemma_column, lemma),
            (self.pos_column, pos),
        )
        constraints = [f'{col}="{pattern}"%c' for col, pattern in columns if pattern]
        return f"[{' & '.join(constraints)}]"

    # --- terminals -----------------------------------------------------

    def WORD(self, token: Any) -> str:
        return self._token(word=_to_regex(str(token)))

    def GAPS(self, token: Any) -> str:
        return _gap_tokens(str(token))

    def POS_TAG(self, token: Any) -> str:
        parts = _POS_TAG_PARTS.fullmatch(str(token))
        assert parts is not None  # lexed from this same shape
        return self._token(
            word=_to_regex(parts["word"]), pos=_resolve_pos_tag(parts["tag"])
        )

    def LEMMA(self, token: Any) -> str:
        """Handle `{lemma}`, `{lemma/CLASS}` and `{lemma}_TAG` alike."""
        parts = _LEMMA_PARTS.fullmatch(str(token))
        assert parts is not None
        lemma, klass = _split_lemma(parts["lemma"])
        # The `/CLASS` slot only ever names a simplified class -- `{walk}_VB*`
        # is how to ask for a tag pattern -- so the two spellings stay distinct.
        if parts["tag"]:
            pos = _resolve_pos_tag(parts["tag"])
        else:
            pos = _pos_class(klass) if klass else ""
        return self._token(lemma=_to_regex(lemma), pos=pos)

    LEMMA_POS_TAG = LEMMA

    # --- rules ---------------------------------------------------------

    def seq(self, items: list[str]) -> str:
        return " ".join(items)

    def group(self, items: list[Any]) -> str:
        *alts, rparen_quant = items
        closing = str(rparen_quant)
        quant = closing[1:] if len(closing) > 1 else ""
        body = "|".join(alts) if len(alts) > 1 else alts[0]
        return f"({body}){quant}"

    def binding(self, items: list[Any]) -> str:
        head = str(items[0])
        var_name = head[1:-1]  # strip `$` and `:`
        return f"${var_name}: ({items[1]})"

    def start(self, items: list[str]) -> str:
        return items[0]


_parser = Lark(_GRAMMAR, start="start", parser="lalr", lexer="basic")


def simple_to_cqp(
    query: str,
    token_column: str = "token",
    pos_column: str = "pos",
    lemma_column: str = "lemma",
) -> str:
    r"""Parse a simple query and convert it to CQP syntax.

    Parameters
    ----------
    query : str
        Simple query string using BNCweb syntax
    token_column : str, optional
        Column name for token searches (default: "token")
    pos_column : str, optional
        Column name for POS tag searches (default: "pos")
    lemma_column : str, optional
        Column name for lemma searches (default: "lemma")

    Returns
    -------
    str
        Equivalent CQP query string

    Raises
    ------
    lark.exceptions.LarkError
        If the query syntax is invalid
    ValueError
        If a `_{CLASS}` tag or a `{lemma/CLASS}` slot does not name a
        simplified POS class

    Examples
    --------
    >>> simple_to_cqp("fox")
    '[token="fox"%c]'

    >>> simple_to_cqp("s?ng")
    '[token="s.ng"%c]'

    >>> simple_to_cqp("*able")
    '[token=".*able"%c]'

    >>> simple_to_cqp("[car,truck]")
    '[token="(?:car|truck)"%c]'

    >>> simple_to_cqp("neighbo[u,]r")
    '[token="neighbo(?:u|)r"%c]'

    >>> simple_to_cqp("quick brown fox")
    '[token="quick"%c] [token="brown"%c] [token="fox"%c]'

    >>> simple_to_cqp("fox + over")
    '[token="fox"%c] []{1} [token="over"%c]'

    >>> simple_to_cqp("fox +* over")
    '[token="fox"%c] []{1,2} [token="over"%c]'

    >>> simple_to_cqp("lights_NN2")
    '[token="lights"%c & pos="NN2"%c]'

    >>> simple_to_cqp("_PNX")
    '[pos="PNX"%c]'

    >>> simple_to_cqp("_{SUBST}")
    '[pos="N.*"%c]'

    >>> simple_to_cqp("{light}")
    '[lemma="light"%c]'

    >>> simple_to_cqp("{light/V}")
    '[lemma="light"%c & pos="V.*"%c]'

    >>> simple_to_cqp("{walk}_VBD")
    '[lemma="walk"%c & pos="VBD"%c]'

    >>> simple_to_cqp("{be}_V*")
    '[lemma="be"%c & pos="V.*"%c]'

    >>> simple_to_cqp("{box}_{SUBST}")
    '[lemma="box"%c & pos="N.*"%c]'

    >>> simple_to_cqp("{[car,truck]}")
    '[lemma="(?:car|truck)"%c]'

    >>> simple_to_cqp(r"{a\}b}_NN")
    '[lemma="a\\}b"%c & pos="NN"%c]'

    >>> simple_to_cqp("$x: fox")
    '$x: ([token="fox"%c])'

    >>> simple_to_cqp("$det: the $noun: fox")
    '$det: ([token="the"%c]) $noun: ([token="fox"%c])'

    >>> simple_to_cqp("$phrase: (quick brown)")
    '$phrase: (([token="quick"%c] [token="brown"%c]))'
    """
    tree = _parser.parse(query)
    compiler = SimpleCompiler(token_column, pos_column, lemma_column)
    try:
        return compiler.transform(tree)
    except VisitError as exc:  # lark wraps the cause; the caller wants it plain
        raise exc.orig_exc from None
