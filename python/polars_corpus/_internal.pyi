from __future__ import annotations

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

class OpcodeMatcher:
    def __init__(self, opcodes: list[Any], masks: list[Any]) -> None: ...
    def matchall(self) -> Optional[list[Match]]: ...
