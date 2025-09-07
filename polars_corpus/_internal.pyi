from __future__ import annotations

from enum import Enum
from typing import Any, Optional

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

class Span:
    def __init__(self, start: int, end: int) -> None: ...

class Opcode(Enum):
    TOKEN = 0
    JUMP = 1
    SPLIT = 2
    SKIP = 3
    MATCH = 4

    def __init__(self) -> None: ...

class OpcodeMatcher:
    def __init__(self, opcodes: list[Any], masks: list[Any]) -> None: ...
    def matchall(self) -> Optional[list[Any]]: ...
