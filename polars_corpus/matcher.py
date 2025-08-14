from __future__ import annotations

from typing import Any, Optional

import polars as pl

from ._internal import Opcode, OpcodeMatcher, Span
from .search import SearchResults
from .cqp_parser import cqp

__all__ = ["search", "Span"]


def col_name(i: int) -> str:
    return f"_{i}"


def compute_masks(df: pl.DataFrame, opcodes: list[Any]) -> pl.DataFrame:
    """Find token positions where a (sub-)query might potentially match"""
    for pc in range(len(opcodes)):
        df = propagate_masks(df, opcodes, pc)
    df = df.select([col_name(i) for i in range(len(opcodes))])
    return df.rechunk()


def propagate_masks(df: pl.DataFrame, opcodes: list[Any], pc: int) -> pl.DataFrame:
    """Propagate token masks backwards through the NFA"""
    if col_name(pc) not in df:
        match opcodes[pc]:
            case (Opcode.TOKEN, expr):
                df = df.with_columns(expr.fill_null(False).alias(col_name(pc)))
            case (Opcode.MATCH,) | (Opcode.SKIP,):
                df = df.with_columns(pl.lit(True).alias(col_name(pc)))
            case (Opcode.JUMP, offset):
                if col_name(pc + offset) not in df.columns:
                    df = propagate_masks(df, opcodes, pc + offset)
                df = df.with_columns(pl.col(col_name(pc + offset)).alias(col_name(pc)))
            case (Opcode.SPLIT, offset1, offset2):
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


def get_matches(df: pl.DataFrame, query: str) -> Optional[list[Span]]:
    """Parse query and retrieve matching spans"""
    if df.is_empty():
        return None

    opcodes = list(cqp.parse_string(query, parse_all=True))
    opcodes.append((Opcode.MATCH,))

    mask_df = compute_masks(df, opcodes)
    masks = [mask_df.get_column(col) for col in mask_df.columns]
    opcode_matcher = OpcodeMatcher(opcodes, masks)

    return opcode_matcher.matchall()


def search(df: pl.DataFrame, query: str) -> Optional[SearchResults]:
    """Search corpus using a CQP-style query.

    Parameters
    ----------
    df : pl.DataFrame
         Corpus to be searched.
    query: str
        Search query

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
    Put in a link to docs about the query language

    """
    if matched_spans := get_matches(df, query):
        return SearchResults(df, query, matched_spans)
    else:
        return None


## CQP parsing logic has been moved to cqp_parser.py
