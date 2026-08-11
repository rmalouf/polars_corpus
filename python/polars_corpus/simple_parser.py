"""Parser for BNCweb-style Simple Query Language.

This module implements a parser for the simple query syntax used in BNCweb,
which provides an alternative to CQP syntax for corpus searches. The parser
translates simple queries directly to CQP expressions.

Supports variable bindings using $varname: pattern syntax, which translates
to CQP's $varname: (pattern) format with automatic parenthesis wrapping.

See simple_grammar.md for the full grammar specification.
"""

from __future__ import annotations

import re
from typing import Any

from lark import Lark, Transformer
from lark.exceptions import VisitError

from .utils import check_choice

__all__ = ["simple_to_cqp"]


# Helper functions for building CQP expressions
def _make_constraint(col: str, pattern: str) -> str:
    """Build a single column constraint. All matching is case-insensitive."""
    return f'{col}="{pattern}"%c'


def _make_token(*constraints: str) -> str:
    """Build a token constraint with one or more conditions."""
    return f"[{' & '.join(constraints)}]"


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

LEMMA_POS_TAG: /\{{{_LI}\}}_(?:\{{[A-Za-z]+\}}|{_PC}+)/
LEMMA: /\{{{_LI}\}}/
// POS_TAG and WORD overlap (e.g. "fox_NN" could start with a WORD match on
// "fox"); give POS_TAG higher priority so Lark's lexer prefers it over the
// shorter WORD match. Longest-match alone isn't reliable here with complex
// character classes in the Lark basic lexer.
POS_TAG.2: /{_PC}*_(?:\{{[A-Za-z]+\}}|{_PC}+)/
GAPS: /[+*]+/
WORD: /{_PC}*{_NW}{_PC}*/

%ignore /[ \t\r\n]+/
"""


# Splits a pattern into escape sequences and single characters.
_PATTERN_PARTS = re.compile(rf"{_ESC}|.", re.S)
# Splits a word into escape sequences, alternative groups, and single characters.
_WORD_PARTS = re.compile(rf"{_ESC}|{_ALT}|.", re.S)
# Whitespace padding an alternative, which is layout rather than part of the
# pattern. An escaped space (`\ `) is a literal and so is left alone.
_ALT_PAD = re.compile(r"^\s+|(?<!\\)\s+$")
# Separators that only separate when unescaped: `,` between alternatives,
# `/` between a lemma and its POS constraint.
_ALT_SEP = re.compile(r"(?<!\\),")
_LEMMA_SEP = re.compile(r"(?<!\\)/")

_WILDCARDS = {"?": ".", "*": ".*", "+": ".+"}


def _literal(char: str) -> str:
    """Regex matching `char` itself, safe to place in a CQP double-quoted value."""
    return r"\"" if char == '"' else re.escape(char)


def wildcard_to_regex(pattern: str) -> str:
    """Convert simple query wildcards to regex pattern.

    Wildcards:
    - ? = single character (.)
    - * = zero or more characters (.*)
    - + = one or more characters (.+)

    A backslash-escaped character is a literal, so `x\\*x` matches `x*x`.
    """
    parts = []
    for part in _PATTERN_PARTS.findall(pattern):
        if len(part) == 2:  # `\X` escape: the escaped character is a literal
            parts.append(_literal(part[1]))
        else:
            parts.append(_WILDCARDS.get(part, _literal(part)))
    return "".join(parts)


def word_to_regex(word: str) -> str:
    """Convert a word pattern to regex, expanding `[a,b]` alternative groups.

    Groups may appear anywhere in the word (`neighbo[u,]r`) and their
    alternatives may be empty or contain wildcards.
    """
    parts = []
    for part in _WORD_PARTS.findall(word):
        if part.startswith("["):
            alts = "|".join(
                wildcard_to_regex(_ALT_PAD.sub("", alt))
                for alt in _ALT_SEP.split(part[1:-1])
            )
            parts.append(f"(?:{alts})")
        else:
            parts.append(wildcard_to_regex(part))
    return "".join(parts)


def _split_lemma(inner: str) -> tuple[str, str]:
    """Split `lemma[/POS]` on the first unescaped slash."""
    parts = _LEMMA_SEP.split(inner, maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _gap_tokens(gap_str: str) -> str:
    plus_count = gap_str.count("+")
    star_count = gap_str.count("*")
    min_tokens = plus_count
    max_tokens = plus_count + star_count

    if min_tokens == max_tokens:
        return f"[]{{{min_tokens}}}"
    if max_tokens == min_tokens + 1:
        if min_tokens == 0:
            return "[]?"
        return f"[]{{{min_tokens}}} []?"
    return f"[]{{{min_tokens},{max_tokens}}}"


def _resolve_pos_tag(raw: str) -> str:
    """Convert a captured POS tag fragment to a CQP pattern.

    `{CLASS}` names a simplified class; anything else is a literal tag pattern,
    even where it spells one of the class names.
    """
    if raw.startswith("{") and raw.endswith("}"):
        return _pos_class(raw[1:-1])
    return word_to_regex(raw)


def _pos_class(name: str) -> str:
    """Expand a simplified POS class name, e.g. `SUBST` -> `N.*`."""
    return _POS_MAPPING[check_choice(name, _POS_CLASSES, param="POS class").upper()]


class SimpleCompiler(Transformer):
    def __init__(self, token_column: str, pos_column: str, lemma_column: str) -> None:
        super().__init__()
        self.token_column = token_column
        self.pos_column = pos_column
        self.lemma_column = lemma_column

    # --- terminals -----------------------------------------------------

    def WORD(self, token: Any) -> str:
        pattern = word_to_regex(str(token))
        return _make_token(_make_constraint(self.token_column, pattern))

    def GAPS(self, token: Any) -> str:
        return _gap_tokens(str(token))

    def POS_TAG(self, token: Any) -> str:
        raw = str(token)
        # Split on the first underscore that introduces the tag. The word part
        # cannot contain a literal `_` (only wildcards/word chars/escapes), and
        # the tag part starts with either `{` or a word/wildcard char.
        idx = self._split_pos(raw)
        word_part = raw[:idx]
        tag_part = raw[idx + 1 :]
        pos_pattern = _resolve_pos_tag(tag_part)
        constraints = [_make_constraint(self.pos_column, pos_pattern)]
        if word_part:
            word_pattern = word_to_regex(word_part)
            constraints.insert(0, _make_constraint(self.token_column, word_pattern))
        return _make_token(*constraints)

    @staticmethod
    def _split_pos(raw: str) -> int:
        # Find the first `_` that isn't inside an escape sequence or `[...]` group.
        i = 0
        while i < len(raw):
            ch = raw[i]
            if ch == "\\" and i + 1 < len(raw):
                i += 2
                continue
            if ch == "[":
                i = raw.index("]", i) + 1
                continue
            if ch == "_":
                return i
            i += 1
        raise ValueError(f"POS_TAG has no unescaped underscore: {raw!r}")

    def LEMMA(self, token: Any) -> str:
        raw = str(token)[1:-1]  # strip braces
        lemma_part, pos_part = _split_lemma(raw)
        lemma_pattern = wildcard_to_regex(lemma_part)
        constraints = [_make_constraint(self.lemma_column, lemma_pattern)]
        if pos_part:
            pos_pattern = _POS_MAPPING.get(pos_part.upper(), pos_part + ".*")
            constraints.append(_make_constraint(self.pos_column, pos_pattern))
        return _make_token(*constraints)

    def LEMMA_POS_TAG(self, token: Any) -> str:
        raw = str(token)
        # Matches `{lemma[/POS]}_TAG` — find the `}_` boundary.
        brace_end = raw.index("}")
        lemma_inner = raw[1:brace_end]
        tag_part = raw[brace_end + 2 :]  # skip `}_`
        lemma_part, _ = _split_lemma(lemma_inner)
        lemma_pattern = wildcard_to_regex(lemma_part)
        pos_pattern = _resolve_pos_tag(tag_part)
        return _make_token(
            _make_constraint(self.lemma_column, lemma_pattern),
            _make_constraint(self.pos_column, pos_pattern),
        )

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
    """Parse a simple query and convert it to CQP syntax.

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
        If a `_{CLASS}` tag does not name a simplified POS class

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
