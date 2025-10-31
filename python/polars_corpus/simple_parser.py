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
    flag = "" if case_sensitive else "%c"
    return f'{col}="{pattern}"{flag}'


def _make_token(*constraints: str) -> str:
    """Build a token constraint with one or more conditions."""
    return f"[{' & '.join(constraints)}]"


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
    result = result.replace(r"\?", ".")
    result = result.replace(r"\*", ".*")
    result = result.replace(r"\+", ".+")

    return result


# Pyparsing grammar
# Parse actions return tuples that will be converted to CQP later

# Simplified POS tag mapping - supports both BNC CLAWS-5 and Penn Treebank tagsets
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


# Define metacharacters that can be escaped
metacharacters = "?*+,:@/()[]{}_ -<>"

# Escaped character: backslash followed by metacharacter
escaped_char = pp.Combine(pp.Literal("\\") + pp.Char(metacharacters)).set_parse_action(
    lambda t: t[0][1]
)  # Remove backslash

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
        pp.Suppress("[")
        + pp.delimited_list(alternative_word, delim=",")
        + pp.Suppress("]")
    )

    def make_alternative(t: pp.ParseResults) -> str:
        patterns = [wildcard_to_regex(alt) for alt in t]
        combined = "|".join(patterns)
        return _make_token(_make_constraint(column, combined))

    alternative_list.set_parse_action(make_alternative)

    # Word token: must contain at least one non-wildcard character
    # This ensures standalone * and + are parsed as gap tokens, not words
    # Patterns:
    # 1. Starts with non-wildcard, followed by anything
    # 2. Starts with wildcard(s), but must have at least one non-wildcard somewhere
    word_with_content = pp.Combine(
        # Pattern 1: non-wildcard start
        (word_char | escaped_char) + pp.ZeroOrMore(word_part)
        |
        # Pattern 2: wildcard(s) followed by at least one non-wildcard
        pp.OneOrMore(wildcard_char)
        + (word_char | escaped_char)
        + pp.ZeroOrMore(word_part)
    )

    def make_word(t: pp.ParseResults) -> str:
        pattern = wildcard_to_regex(t[0])
        return _make_token(_make_constraint(column, pattern))

    word_with_content.set_parse_action(make_word)

    # Lemma pattern: {lemma} or {lemma/POS}
    # Lemma part can include wildcards, optional /POS suffix
    lemma_word_part = pp.Combine(pp.OneOrMore(word_part))
    lemma_pos_part = pp.Combine(
        pp.OneOrMore(pp.Char(pp.alphas))
    )  # Simplified POS tags are alpha only
    lemma_only_pattern = (
        pp.Suppress("{")
        + lemma_word_part
        + pp.Optional(pp.Suppress("/") + lemma_pos_part)
        + pp.Suppress("}")
    )
    lemma_pattern = lemma_only_pattern.copy()

    def make_lemma(t: pp.ParseResults) -> str:
        lemma_part = t[0]
        pos_part = t[1] if len(t) > 1 else None
        lemma_pattern = wildcard_to_regex(lemma_part)
        constraints = [_make_constraint(lemma_column, lemma_pattern)]
        if pos_part:
            pos_pattern = _POS_MAPPING.get(pos_part.upper(), pos_part + ".*")
            constraints.append(
                _make_constraint(pos_column, pos_pattern, case_sensitive=True)
            )
        return _make_token(*constraints)

    lemma_pattern.set_parse_action(make_lemma)

    # Define pos_word_part_item for use in both patterns
    pos_word_part_item = (
        escaped_char | wildcard_char | pp.Char(pp.alphas + pp.nums + "!@#$%^&=\\-")
    )

    # Lemma+POS pattern: {lemma}_TAG or {lemma}_{SIMPLIFIED}
    # Supports both exact POS tags (e.g., {walk}_VBD) and simplified tags (e.g., {walk}_{SUBST})
    # Simplified tags are wrapped in braces and expanded using _POS_MAPPING
    simplified_pos_tag = (
        pp.Suppress("{")
        + pp.Combine(pp.OneOrMore(pp.Char(pp.alphas)))
        + pp.Suppress("}")
    )
    exact_pos_tag = pp.Combine(pp.OneOrMore(pos_word_part_item))

    lemma_pos_tag_pattern = (
        lemma_only_pattern + pp.Suppress("_") + (simplified_pos_tag | exact_pos_tag)
    )

    def make_lemma_pos_tag(t: pp.ParseResults) -> str:
        lemma_part = t[0]
        pos_part = t[-1]
        lemma_pattern = wildcard_to_regex(lemma_part)

        # Check if this is a simplified POS tag (would have been parsed from {TAG})
        # We detect this by checking if pos_part is all alpha and matches a key in _POS_MAPPING
        if pos_part.upper() in _POS_MAPPING:
            # It's a simplified tag - expand it
            pos_pattern = _POS_MAPPING[pos_part.upper()]
        else:
            # It's an exact tag - convert wildcards
            pos_pattern = wildcard_to_regex(pos_part)

        return _make_token(
            _make_constraint(lemma_column, lemma_pattern),
            _make_constraint(pos_column, pos_pattern, case_sensitive=True),
        )

    lemma_pos_tag_pattern.set_parse_action(make_lemma_pos_tag)

    # POS tag pattern: word_TAG, _TAG, word_{TAG}, or _{TAG}
    # Word part is optional (for _TAG pattern), POS part after underscore
    # POS part can optionally be wrapped in braces for simplified tags
    # Important: Use Combine to prevent consuming whitespace between elements
    pos_word_char = pp.Char(pp.alphas + pp.nums + "!@#$%^&=\\-")
    pos_word_part_item = escaped_char | wildcard_char | pos_word_char
    pos_word_part_content = pp.ZeroOrMore(pos_word_part_item)

    # POS tag can be: {TAG} (simplified, in braces) or TAG (exact/wildcard)
    braced_pos_tag = (
        pp.Suppress("{")
        + pp.Combine(pp.OneOrMore(pp.Char(pp.alphas)))
        + pp.Suppress("}")
    )
    unbraced_pos_tag = pp.Combine(pp.OneOrMore(pos_word_part_item))

    # Combine the entire pattern so it doesn't consume whitespace
    pos_pattern = pp.Combine(pos_word_part_content + pp.Literal("_")) + (
        braced_pos_tag | unbraced_pos_tag
    )

    def make_pos_tag(t: pp.ParseResults) -> str:
        # t[0] contains "word_" (with trailing underscore)
        # t[1] contains the POS tag (with or without braces)
        word_with_underscore = t[0]
        word_part = word_with_underscore[:-1]  # Remove trailing underscore
        pos_part = t[1]

        # Check if this was a braced tag by seeing if it's all alpha and in mapping
        # (braced_pos_tag only matches alpha characters)
        is_braced = pos_part.isalpha() and pos_part.upper() in _POS_MAPPING

        if is_braced:
            # Simplified tag in braces - expand using mapping
            pos_pattern = _POS_MAPPING[pos_part.upper()]
        else:
            # Exact tag or wildcard pattern
            pos_pattern = wildcard_to_regex(pos_part)

        constraints = [_make_constraint(pos_column, pos_pattern, case_sensitive=True)]
        if word_part:
            word_pattern = wildcard_to_regex(word_part)
            constraints.insert(0, _make_constraint(column, word_pattern))
        return _make_token(*constraints)

    pos_pattern.set_parse_action(make_pos_tag)

    # Gap tokens - consecutive * or + characters, standalone (not part of a word)
    # Multiple consecutive + or * represent multiple gaps
    # Examples: ++ = 2 tokens, *** = 0-3 tokens, +++** = 3-5 tokens
    # But: *able, +able, **oom should be parsed as word patterns, not gaps
    # Solution: Match gap chars followed by whitespace, end, or special chars (not word chars)
    # The negative lookahead ensures gaps aren't followed by word-forming characters
    # Note: We exclude * and + from the negative lookahead so **oom is treated as a word
    consecutive_gaps = pp.Regex(r"[+*]+(?![a-zA-Z0-9!@#$%^&=\\\-*+?])")

    def make_consecutive_gaps(t: pp.ParseResults) -> str:
        gap_str = t[0]
        plus_count = gap_str.count("+")
        star_count = gap_str.count("*")

        # Calculate min and max tokens
        min_tokens = plus_count  # Each + requires one token
        max_tokens = plus_count + star_count  # Each * adds an optional token

        if min_tokens == max_tokens:
            # Exact count
            return f"[]{{{min_tokens}}}"
        elif max_tokens == min_tokens + 1:
            # Single optional token - use ? for efficiency
            if min_tokens == 0:
                return "[]?"
            else:
                return f"[]{{{min_tokens}}}" + " []?"
        else:
            # Range of tokens
            return f"[]{{{min_tokens},{max_tokens}}}"

    consecutive_gaps.set_parse_action(make_consecutive_gaps)

    # Base sequence item (without groups/quantifiers): lemma+POS, lemma, gaps, POS, alternative, or word
    base_item = (
        lemma_pos_tag_pattern
        | lemma_pattern
        | consecutive_gaps
        | pos_pattern
        | alternative_list
        | word_with_content
    )

    # Forward declaration for recursive grammar (groups can contain sequences)
    sequence_item = pp.Forward()

    # Quantifiers for groups: ?, +, *, {n}, {m,n}
    # Note: The simple literals must come before the regex to avoid ambiguity
    quantifier = (
        pp.Literal("?")
        | pp.Literal("+")
        | pp.Literal("*")
        | pp.Regex(r"\{\d+,\d+\}")
        | pp.Regex(r"\{\d+\}")
    )

    # Group: (sequence) or (alternative1 | alternative2 | ...) with optional quantifier
    # Each alternative is a sequence of items
    # Disjunction (pipe-separated alternatives) is supported
    group_sequence = pp.Group(pp.OneOrMore(sequence_item))
    group_content = group_sequence + pp.ZeroOrMore(pp.Suppress("|") + group_sequence)
    group_pattern = (
        pp.Suppress("(")
        + pp.Group(group_content)
        + pp.Suppress(")")
        + pp.Optional(quantifier)
    )

    def make_group(t: pp.ParseResults) -> str:
        content = t[0]  # The group content (may contain multiple alternatives)
        quant = t[1] if len(t) > 1 else None

        # Check if we have disjunction (multiple alternatives)
        # content is a list of sequences (each sequence is a list of items)
        if len(content) > 1:
            # Multiple alternatives - join each sequence and then join with |
            alternatives = []
            for sequence in content:
                sequence_cqp = " ".join(sequence)
                alternatives.append(sequence_cqp)
            result_cqp = "|".join(alternatives)
        else:
            # Single sequence - just join the items
            sequence_cqp = " ".join(content[0])
            result_cqp = sequence_cqp

        # Wrap in parentheses and add quantifier if present
        if quant:
            return f"({result_cqp}){quant}"
        else:
            return f"({result_cqp})"

    group_pattern.set_parse_action(make_group)

    # A sequence item is: group or base_item
    # Groups must come before base items to be matched first
    sequence_item <<= group_pattern | base_item

    # A query is a sequence of items
    return pp.OneOrMore(sequence_item)


def simple_to_cqp(
    query: str,
    column: str = "token",
    pos_column: str = "pos",
    lemma_column: str = "lemma",
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

    >>> simple_to_cqp("{box}_{SUBST}")
    '[lemma="box"%c & pos="N.*"]'
    """
    # Build grammar with parse actions for the specified columns
    grammar = _build_grammar(column, pos_column, lemma_column)

    # Parse the query - parse actions generate CQP directly
    cqp_tokens = grammar.parse_string(query, parse_all=True)

    return " ".join(cqp_tokens)
