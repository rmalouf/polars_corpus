"""Parser for BNCweb-style Simple Query Language.

This module implements a parser for the simple query syntax used in BNCweb,
which provides an alternative to CQP syntax for corpus searches. The parser
translates simple queries directly to CQP expressions.

See simple_grammar.md for the full grammar specification.
"""

from __future__ import annotations

import re

import pyparsing as pp

__all__ = ["simple_to_cqp"]


# Helper functions for building CQP expressions
def _make_constraint(col: str, pattern: str, case_sensitive: bool = False) -> str:
    """Build a single column constraint."""
    flag = '' if case_sensitive else '%c'
    return f'{col}="{pattern}"{flag}'


def _make_token(*constraints: str) -> str:
    """Build a token constraint with one or more conditions."""
    return f'[{" & ".join(constraints)}]'


def wildcard_to_regex(pattern: str) -> str:
    """Convert simple query wildcards to regex pattern.

    Wildcards:
    - ? = single character (.)
    - * = zero or more characters (.*)
    - + = one or more characters (.+)
    """
    # First escape all regex metacharacters
    result = re.escape(pattern)

    # Then replace escaped wildcards with regex equivalents
    # re.escape will have turned ? into \?, * into \*, + into \+
    result = result.replace(r'\?', '.')
    result = result.replace(r'\*', '.*')
    result = result.replace(r'\+', '.+')

    return result


# Pyparsing grammar
# Parse actions return tuples that will be converted to CQP later

# Simplified POS tag mapping - supports both BNC CLAWS-5 and Penn Treebank tagsets
_POS_MAPPING = {
    'V': 'V.*', 'VERB': 'V.*',
    'N': 'N.*', 'SUBST': 'N.*',
    'A': '(AJ.*|JJ.*)', 'ADJ': '(AJ.*|JJ.*)',
    'ADV': '(AV.*|RB.*)',
    'ART': '(AT.*|DT)', 'CONJ': '(CJ.*|CC)',
    'PREP': '(PR.*|IN|TO)',
    'PRON': '(PN.*|PRP.*)',
    'INT': '(ITJ|UH)', 'INTERJ': '(ITJ|UH)',
    'STOP': 'PU.*',
    'UNC': 'UNC'
}


# Define metacharacters that can be escaped
metacharacters = "?*+,:@/()[]{}_ -<>"

# Escaped character: backslash followed by metacharacter
escaped_char = pp.Combine(
    pp.Literal("\\") + pp.Char(metacharacters)
).set_parse_action(lambda t: t[0][1])  # Remove backslash

# Regular characters for words (not wildcards or special chars)
# Note: underscore is NOT included here because it's used for POS patterns
word_char = pp.Char(pp.alphas + pp.nums + "!@#$%^&=\\-")

# Wildcard characters
wildcard_char = pp.Char("?*+")

# Character parts that can appear in words or alternatives
word_part = escaped_char | wildcard_char | word_char

def _build_grammar(column: str, pos_column: str, lemma_column: str) -> pp.ParserElement:
    """Build grammar with parse actions that generate CQP directly."""

    # Alternative list: [alt1,alt2,alt3] or [u,] for optional
    alternative_word = pp.Combine(pp.ZeroOrMore(word_part))
    alternative_list = (
        pp.Suppress("[") +
        pp.delimited_list(alternative_word, delim=",") +
        pp.Suppress("]")
    )
    def make_alternative(t):
        patterns = [wildcard_to_regex(alt) for alt in t]
        combined = "|".join(patterns)
        return _make_token(_make_constraint(column, combined))
    alternative_list.set_parse_action(make_alternative)

    # Word token: must contain at least one non-wildcard character
    # This ensures standalone * and + are parsed as gap tokens, not words
    word_with_content = pp.Combine(
        (word_char | escaped_char) + pp.ZeroOrMore(word_part) |
        wildcard_char + pp.OneOrMore(word_part)
    )
    def make_word(t):
        pattern = wildcard_to_regex(t[0])
        return _make_token(_make_constraint(column, pattern))
    word_with_content.set_parse_action(make_word)

    # Lemma pattern: {lemma} or {lemma/POS}
    # Lemma part can include wildcards, optional /POS suffix
    lemma_word_part = pp.Combine(pp.OneOrMore(word_part))
    lemma_pos_part = pp.Combine(pp.OneOrMore(pp.Char(pp.alphas)))  # Simplified POS tags are alpha only
    lemma_only_pattern = (
        pp.Suppress("{") +
        lemma_word_part +
        pp.Optional(pp.Suppress("/") + lemma_pos_part) +
        pp.Suppress("}")
    )
    lemma_pattern = lemma_only_pattern.copy()
    def make_lemma(t):
        lemma_part = t[0]
        pos_part = t[1] if len(t) > 1 else None
        lemma_pattern = wildcard_to_regex(lemma_part)
        constraints = [_make_constraint(lemma_column, lemma_pattern)]
        if pos_part:
            pos_pattern = _POS_MAPPING.get(pos_part.upper(), pos_part + '.*')
            constraints.append(_make_constraint(pos_column, pos_pattern, case_sensitive=True))
        return _make_token(*constraints)
    lemma_pattern.set_parse_action(make_lemma)

    # Define pos_word_part_item for use in both patterns
    pos_word_part_item = escaped_char | wildcard_char | pp.Char(pp.alphas + pp.nums + "!@#$%^&=\\-")

    # Lemma+POS pattern: {lemma}_TAG
    # This is for exact POS tags (not simplified), e.g., {walk}_VBD
    lemma_pos_tag_pattern = (
        lemma_only_pattern +
        pp.Suppress("_") +
        pp.Combine(pp.OneOrMore(pos_word_part_item))
    )
    def make_lemma_pos_tag(t):
        lemma_part = t[0]
        pos_part = t[-1]
        lemma_pattern = wildcard_to_regex(lemma_part)
        pos_pattern = wildcard_to_regex(pos_part)
        return _make_token(
            _make_constraint(lemma_column, lemma_pattern),
            _make_constraint(pos_column, pos_pattern, case_sensitive=True)
        )
    lemma_pos_tag_pattern.set_parse_action(make_lemma_pos_tag)

    # POS tag pattern: word_TAG or _TAG
    # Word part is optional (for _TAG pattern), POS part after underscore
    # Important: Use Combine to prevent consuming whitespace between elements
    pos_word_char = pp.Char(pp.alphas + pp.nums + "!@#$%^&=\\-")
    pos_word_part_item = escaped_char | wildcard_char | pos_word_char
    pos_word_part_content = pp.ZeroOrMore(pos_word_part_item)
    pos_tag_part_content = pp.OneOrMore(pos_word_part_item)

    # Combine the entire pattern so it doesn't consume whitespace
    pos_pattern = pp.Combine(
        pos_word_part_content + pp.Literal("_") + pos_tag_part_content
    )
    def make_pos_tag(t):
        word_part, pos_part = t[0].split('_')
        pos_pattern = wildcard_to_regex(pos_part)
        constraints = [_make_constraint(pos_column, pos_pattern, case_sensitive=True)]
        if word_part:
            word_pattern = wildcard_to_regex(word_part)
            constraints.insert(0, _make_constraint(column, word_pattern))
        return _make_token(*constraints)
    pos_pattern.set_parse_action(make_pos_tag)

    # Gap tokens - single * or + (standalone)
    gap_plus = pp.Literal("+").set_parse_action(lambda: '[]+')
    gap_star = pp.Literal("*").set_parse_action(lambda: '[]?')

    # A sequence item is: lemma+POS pattern, lemma pattern, POS pattern, alternative, word, or gap
    # Order matters: try most specific patterns first
    sequence_item = lemma_pos_tag_pattern | lemma_pattern | pos_pattern | alternative_list | word_with_content | gap_star | gap_plus

    # A query is a sequence of items
    return pp.OneOrMore(sequence_item)


def simple_to_cqp(
    query: str,
    column: str = "token",
    pos_column: str = "pos",
    lemma_column: str = "lemma"
) -> str:
    """Parse a simple query and convert it to CQP syntax.

    Parameters
    ----------
    query : str
        Simple query string using BNCweb syntax
    column : str, optional
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
    ParseException
        If the query syntax is invalid

    Examples
    --------
    >>> simple_to_cqp("fox")
    '[token="fox"%c]'

    >>> simple_to_cqp("s?ng")
    '[token="s.ng"%c]'

    >>> simple_to_cqp("*able")
    '[token=".*able"%c]'

    >>> simple_to_cqp("[car,truck]")
    '[token="car|truck"%c]'

    >>> simple_to_cqp("quick brown fox")
    '[token="quick"%c] [token="brown"%c] [token="fox"%c]'

    >>> simple_to_cqp("fox + over")
    '[token="fox"%c] []+ [token="over"%c]'

    >>> simple_to_cqp("lights_NN2")
    '[token="lights"%c & pos="NN2"]'

    >>> simple_to_cqp("_PNX")
    '[pos="PNX"]'

    >>> simple_to_cqp("{light}")
    '[lemma="light"%c]'

    >>> simple_to_cqp("{light/V}")
    '[lemma="light"%c & pos="V.*"]'

    >>> simple_to_cqp("{walk}_VBD")
    '[lemma="walk"%c & pos="VBD"]'

    >>> simple_to_cqp("{be}_V*")
    '[lemma="be"%c & pos="V.*"]'
    """
    # Build grammar with parse actions for the specified columns
    grammar = _build_grammar(column, pos_column, lemma_column)

    # Parse the query - parse actions generate CQP directly
    cqp_tokens = grammar.parse_string(query, parse_all=True)

    return ' '.join(cqp_tokens)
