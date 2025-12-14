from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import polars as pl

def py_concordance(
    df: pl.DataFrame,
    matched_spans: list[Span],
    chunk_tag: pl.Series,
) -> pl.DataFrame: ...
def py_kwic(
    df: pl.DataFrame,
    matched_spans: list[Span],
    left_window: int,
    right_window: int,
) -> pl.DataFrame: ...
def spans_to_chunks(spans: list[Span], n: int) -> pl.Series: ...

class Span:
    def __init__(self, start: int, end: int) -> None: ...

class Match:
    span: Span
    bindings: Dict[str, Span]
    def __init__(self, span: Span, bindings: Dict[str, Span]) -> None: ...

# TODO: get this from rust instead of redefining it
class Opcode(Enum):
    TOKEN = 0
    JUMP = 1
    SPLIT = 2
    SKIP = 3
    MATCH = 4
    PUSHVAR = 5
    BINDVAR = 6

    def __init__(self) -> None: ...

class OpcodeMatcher:
    def __init__(self, opcodes: list[Any], masks: list[Any]) -> None: ...
    def matchall(self) -> Optional[list[Match]]: ...
