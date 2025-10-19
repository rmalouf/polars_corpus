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
# Parse actions build CQP strings directly - no AST needed

def make_word_cqp(column):
    """Return a parse action that converts a word pattern to CQP token constraint."""
    def action(tokens):
        pattern = wildcard_to_regex(tokens[0])
        return f'[{column}="{pattern}"%c]'
    return action


def make_alternative_cqp(column):
    """Return a parse action that converts alternative list to CQP token constraint."""
    def action(tokens):
        patterns = [wildcard_to_regex(alt) for alt in tokens]
        combined = "|".join(patterns)
        return f'[{column}="{combined}"%c]'
    return action


def make_gap_cqp(gap_type):
    """Return a parse action that converts gap token to CQP."""
    return lambda: '[]?' if gap_type == '*' else '[]+'


def build_grammar(column: str = "token"):
    """Build the pyparsing grammar with the specified column name."""

    # Define metacharacters that can be escaped
    metacharacters = "?*+,:@/()[]{}_ -<>"

    # Escaped character: backslash followed by metacharacter
    escaped_char = pp.Combine(
        pp.Literal("\\") + pp.Char(metacharacters)
    ).set_parse_action(lambda t: t[0][1])  # Remove backslash

    # Regular characters for words (not wildcards or special chars)
    word_char = pp.Char(pp.alphas + pp.nums + "!@#$%^&_=\\-")

    # Wildcard characters
    wildcard_char = pp.Char("?*+")

    # Character parts that can appear in words or alternatives
    word_part = escaped_char | wildcard_char | word_char

    # Alternative list: [alt1,alt2,alt3] or [u,] for optional
    alternative_word = pp.Combine(pp.ZeroOrMore(word_part))
    alternative_list = (
        pp.Suppress("[") +
        pp.delimited_list(alternative_word, delim=",") +
        pp.Suppress("]")
    ).set_parse_action(make_alternative_cqp(column))

    # Word token: must contain at least one non-wildcard character
    # This ensures standalone * and + are parsed as gap tokens, not words
    word_with_content = pp.Combine(
        (word_char | escaped_char) + pp.ZeroOrMore(word_part) |
        wildcard_char + pp.OneOrMore(word_part)
    ).set_parse_action(make_word_cqp(column))

    # Gap tokens - single * or + (standalone)
    gap_plus = pp.Literal("+").set_parse_action(make_gap_cqp('+'))
    gap_star = pp.Literal("*").set_parse_action(make_gap_cqp('*'))

    # A sequence item is either an alternative, word, or gap
    # Order matters: try alternative first, then word (which requires content), then gaps
    sequence_item = alternative_list | word_with_content | gap_star | gap_plus

    # A query is a sequence of items - join with spaces
    return pp.OneOrMore(sequence_item).set_parse_action(lambda t: ' '.join(t))


def simple_to_cqp(query: str, column: str = "token") -> str:
    """Parse a simple query and convert it to CQP syntax.

    Parameters
    ----------
    query : str
        Simple query string using BNCweb syntax
    column : str, optional
        Column name to search (default: "token")

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
    """
    grammar = build_grammar(column)
    parsed = grammar.parse_string(query, parse_all=True)
    return parsed[0]
