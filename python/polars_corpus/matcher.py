from __future__ import annotations

from typing import Optional

import polars as pl

from ._internal import Match, Opcode, OpcodeMatcher, Span
from .cqp_parser import cqp
from .search import SearchResults
from .simple_parser import simple_to_cqp

__all__ = ["search", "search_cqp", "Match", "Span"]


def col_name(i: int) -> str:
    return f"_{i}"


def compute_masks(df: pl.DataFrame, opcodes: list[Opcode]) -> pl.DataFrame:
    """Find token positions where a (sub-)query might potentially match"""
    for pc in range(len(opcodes)):
        df = propagate_masks(df, opcodes, pc)
    df = df.select([col_name(i) for i in range(len(opcodes))])
    return df.rechunk()


def check_bindings(opcodes: list[Opcode]):
    seen: set[str] = set()
    for op in opcodes:
        if isinstance(op, Opcode.BindVar):
            if op._0 in seen:
                raise ValueError(f"Duplicate variable binding: ${op._0}")
            seen.add(op._0)


def propagate_masks(df: pl.DataFrame, opcodes: list[Opcode], pc: int) -> pl.DataFrame:
    """Propagate token masks backwards through the NFA"""
    if col_name(pc) not in df:
        match opcodes[pc]:
            case Opcode.Token(expr):
                expr = pl.Expr.deserialize(expr)
                df = df.with_columns(expr.fill_null(False).alias(col_name(pc)))
            case Opcode.Match() | Opcode.Skip() | Opcode.Fail():
                df = df.with_columns(pl.lit(True).alias(col_name(pc)))
            case (
                Opcode.PushVar()
                | Opcode.PopVar()
                | Opcode.BindVar(_)
                | Opcode.UnBindVar()
            ):
                df = propagate_masks(df, opcodes, pc + 1)
                df = df.with_columns(pl.col(col_name(pc + 1)).alias(col_name(pc)))
            case Opcode.Jump(offset):
                if col_name(pc + offset) not in df.columns:
                    df = propagate_masks(df, opcodes, pc + offset)
                df = df.with_columns(pl.col(col_name(pc + offset)).alias(col_name(pc)))
            case Opcode.Split(offset1, offset2):
                if col_name(pc + offset1) not in df.columns:
                    df = propagate_masks(df, opcodes, pc + offset1)
                if col_name(pc + offset2) not in df.columns:
                    df = propagate_masks(df, opcodes, pc + offset2)
                df = df.with_columns(
                    (
                        pl.col(col_name(pc + offset1)) | pl.col(col_name(pc + offset2))
                    ).alias(col_name(pc))
                )
            case _:
                raise RuntimeError(f"Unknown opcode {opcodes[pc]}")
    return df


def get_matches(df: pl.DataFrame, query: str) -> Optional[list[Match]]:
    """Parse query and retrieve matching spans"""
    if df.is_empty():
        return None

    opcodes = cqp(query)
    opcodes.append(Opcode.Match())

    check_bindings(opcodes)
    mask_df = compute_masks(df, opcodes)
    masks = [mask_df.get_column(col) for col in mask_df.columns]
    opcode_matcher = OpcodeMatcher(opcodes, masks)

    return opcode_matcher.matchall()


def search_cqp(df: pl.DataFrame, query: str) -> Optional[SearchResults]:
    """Search corpus using a CQP-style query.

    Parameters
    ----------
    df : pl.DataFrame
         Corpus to be searched.
    query: str
        CQP query string

    Returns
    -------
    SearchResults
        Result of the search

    Raises
    ------
    ParseException
        If there's an error in the query
    RuntimeError
        If there's an internal error in the search procedure

    Notes
    -----
    For CQP query syntax documentation, see the CQP documentation.

    Examples
    --------
    >>> search_cqp(corpus, '[word="fox"]')
    >>> search_cqp(corpus, '[pos="NN.*"]+ [pos="VB.*"]')

    """
    if matched_spans := get_matches(df, query):
        return SearchResults(df, query, matched_spans)
    else:
        return None


def search(
    df: pl.DataFrame,
    query: str,
    column: str = "token",
    pos_column: str = "pos",
    lemma_column: str = "lemma",
) -> Optional[SearchResults]:
    """Search corpus using simple query language (BNCweb-style).

    This function uses the simple query syntax similar to BNCweb, which is
    more intuitive than CQP for basic searches. Queries are case-insensitive
    by default and support wildcards, alternatives, word sequences, POS tags,
    and lemma searches.

    Parameters
    ----------
    df : pl.DataFrame
         Corpus to be searched.
    query : str
        Simple query string (BNCweb-style syntax)
    column : str, optional
        Column name for token searches (default: "token")
    pos_column : str, optional
        Column name for POS tag searches (default: "pos")
    lemma_column : str, optional
        Column name for lemma searches (default: "lemma")

    Returns
    -------
    SearchResults or None
        Result of the search, or None if no matches found

    Raises
    ------
    ParseException
        If there's an error in the query syntax
    RuntimeError
        If there's an internal error in the search procedure

    Notes
    -----
    Simple query syntax supports:

    - **Basic words**: `fox` matches "fox", "Fox", "FOX" (case-insensitive)
    - **Wildcards**:
      - `?` for single character: `s?ng` → sing, sang, song
      - `*` for zero or more: `*able` → able, table, capable
      - `+` for one or more: `+able` → table, capable (not "able")
    - **Alternatives**: `[car,truck]`, `neighbo[u,]r`
    - **Sequences**: `quick brown fox`
    - **Gaps**:
      - `*` for optional token: `eat * up` → "eat up", "eat it up"
      - `+` for required token: `eat + up` → "eat it up" (not "eat up")
    - **POS tags**: `word_TAG` for word+POS, `_TAG` for POS only
      - `lights_NN2` → "lights" tagged as NN2
      - `*ly_AJ0` → adjectives ending in "-ly"
      - `_PNX` → any reflexive pronoun
    - **Lemmas**: `{lemma}`, `{lemma/POS}`, or `{lemma}_TAG` for lemma searches
      - `{light}` → all forms of "light"
      - `{light/V}` → verbal forms using simplified POS (V, N, A, etc.)
      - `{walk}_VBD` → lemma "walk" with exact POS tag VBD
      - `{be}_V*` → lemma "be" with any verb POS tag
      - `{eat} * up` → lemma "eat" followed by "up"
    - **Variable bindings**: `$varname: pattern` to capture subpatterns
      - `$target: fox` → capture "fox" position
      - `$phrase: (quick brown)` → capture multi-token span
      - Access via `match.bindings[varname]`
    - **Escaping**: `\\?` for literal question mark

    For CQP queries with advanced features, use `search_cqp()` instead.

    Examples
    --------
    >>> search(corpus, "fox")  # Find "fox" (case-insensitive)
    >>> search(corpus, "s?ng")  # Find sing, sang, song
    >>> search(corpus, "*able")  # Find words ending in "able"
    >>> search(corpus, "[car,truck]")  # Find either "car" or "truck"
    >>> search(corpus, "quick brown fox")  # Find exact sequence
    >>> search(corpus, "fox + over")  # "fox" followed by any word, then "over"

    >>> # POS tag searches
    >>> search(corpus, "lights_NN2")  # "lights" as plural noun
    >>> search(corpus, "*ly_AJ0")  # Adjectives ending in "-ly"
    >>> search(corpus, "_PNX")  # Any reflexive pronoun

    >>> # Lemma searches
    >>> search(corpus, "{light}")  # All forms of lemma "light"
    >>> search(corpus, "{light/V}")  # Verbal forms (simplified POS)
    >>> search(corpus, "{walk}_VBD")  # Lemma "walk" with exact POS tag
    >>> search(corpus, "{eat} * up")  # Lemma "eat" followed by "up"

    >>> # Variable bindings
    >>> results = search(corpus, "$verb: {walk}")
    >>> match = results._matches[0]
    >>> verb_span = match.bindings["verb"]
    >>> verb_text = corpus["token"][verb_span.start:verb_span.end]

    >>> # Search in a different column
    >>> search(corpus, "NN*", column="pos")  # Find noun POS tags

    """
    # Translate simple query to CQP
    cqp_query = simple_to_cqp(query, column, pos_column, lemma_column)

    # Use the CQP search function
    if matched_spans := get_matches(df, cqp_query):
        return SearchResults(df, query, matched_spans)
    else:
        return None


## CQP parsing logic has been moved to cqp_parser.py
