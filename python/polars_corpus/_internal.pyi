from __future__ import annotations

from typing import Iterator

import polars as pl

def py_concordance(
    df: pl.DataFrame,
    matches: list[Match],
    chunk_tag: pl.Series,
) -> pl.DataFrame: ...
def py_kwic(
    df: pl.DataFrame,
    matches: list[Match],
    left_window: int,
    right_window: int,
) -> pl.DataFrame: ...
def spans_to_chunks(spans: list[Span], n: int) -> pl.Series: ...

class Span:
    start: int
    end: int
    def __init__(self, start: int, end: int) -> None: ...
    def __repr__(self) -> str: ...
    def __getitem__(self, index: int) -> int: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class Match:
    span: Span
    bindings: dict[str, Span]
    def __init__(self, span: Span, bindings: dict[str, Span]) -> None: ...

class Opcode:
    def __iter__(self) -> Iterator[object]: ...

    class Token(Opcode):
        _0: bytes
        __match_args__ = ("_0",)
        def __init__(self, mask: bytes) -> None: ...

    class Skip(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

    class Jump(Opcode):
        _0: int
        __match_args__ = ("_0",)
        def __init__(self, offset: int) -> None: ...

    class Split(Opcode):
        _0: int
        _1: int
        __match_args__ = ("_0", "_1")
        def __init__(self, offset1: int, offset2: int) -> None: ...

    class Match(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

    class PushVar(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

    class PopVar(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

    class BindVar(Opcode):
        _0: str
        __match_args__ = ("_0",)
        def __init__(self, name: str) -> None: ...

    class UnBindVar(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

    class Fail(Opcode):
        __match_args__ = ()
        def __init__(self) -> None: ...

class OpcodeMatcher:
    def __init__(self, opcodes: list[Opcode], masks: list[pl.Series]) -> None: ...
    def matchall(self) -> list[Match] | None: ...
