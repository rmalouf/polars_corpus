from __future__ import annotations

from collections import namedtuple
from typing import Callable, Iterator, Optional

import numpy as np
import polars as pl
import pyparsing as pp
from tqdm import tqdm

## TODO:
##  1. implement {n,m}
##  2. case insensitive matching
##  3. != and ! in token expressions
##  4. improved valid_starts when the first thing can match the empty string


class ScanContext:
    __slots__ = "max", "vars", "bindings", "trace"

    # TODO: get rid of this class. pass everything as args so we can use local variables

    def __init__(self) -> None:
        pass
        # self.vars: list[Var] = list()
        # self.bindings: dict[int, Any] = dict()


Match = namedtuple("Match", ["start", "end"])


class Pattern:
    def __init__(self) -> None:
        self.subject: Optional[pl.DataFrame] = None
        self.n: int = 0
        self.valid_starts: Optional[np.typing.NDArray[np.bool_]] = None
        self.subpatterns: list[Pattern] = []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(str(p) for p in self.subpatterns)})"

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        raise NotImplementedError

    def set_subject(self, subject: pl.DataFrame) -> None:
        self.subject = subject
        self.n = len(subject)
        for subpattern in self.subpatterns:
            subpattern.set_subject(subject)
        if len(self.subpatterns) > 0:
            self.valid_starts = self.subpatterns[0].valid_starts
        else:
            self.valid_starts = None

    def matchall(
        self: Pattern,
        subject: pl.DataFrame,
        longest_match: bool = True,
        progress: bool = False,
    ) -> Iterator[pl.DataFrame]:
        self.set_subject(subject)
        ctxt = ScanContext()

        if progress:
            bar = tqdm(total=self.n)

        if self.valid_starts is None:
            s = 0
            while s < self.n:
                s0 = s
                longest = 0
                for e in self._op(ctxt, s):
                    if e > longest:
                        longest = e
                if longest > s:
                    yield subject[s:longest]
                    s = longest
                else:
                    s = s + 1
                if progress:
                    bar.update(s - s0)
        else:
            valid_indices = np.where(self.valid_starts)[0]
            i = 0
            s0 = 0
            while i < len(valid_indices):
                s = valid_indices[i]
                if progress:
                    # print(s-s0)
                    bar.update(s - s0)
                s0 = s
                longest = s
                for e in self._op(ctxt, s):
                    if e > longest:
                        longest = e
                    if not longest_match:
                        break
                if longest > s:
                    yield subject[s:longest]
                i = i + 1
            if progress:
                bar.update(self.n - longest)

        if progress:
            bar.close()


class Token(Pattern):
    def __init__(self, constraint: pl.Expr) -> None:
        super().__init__()
        self.constraint = constraint
        # self.valid_tokens: Optional[np.typing.NDArray[np.bool_]] = None

    def __repr__(self) -> str:
        return f'Token({self.constraint}")'

    def set_subject(self, subject: pl.DataFrame) -> None:
        super().set_subject(subject)
        self.valid_tokens: np.typing.NDArray[np.bool_] = (
            subject.select(self.constraint).to_numpy().flatten()
        )
        self.valid_starts = self.valid_tokens

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        if self.valid_tokens[cursor]:
            yield cursor + 1


class Skip(Pattern):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return 'Skip()")'

    def set_subject(self, subject: pl.DataFrame) -> None:
        super().set_subject(subject)
        self.valid_starts = None

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        yield cursor + 1


class MToN(Pattern):
    def __init__(self, pattern: Pattern, m: int = 0, n: Optional[int] = None) -> None:
        super().__init__()
        self.subpatterns = [pattern]
        self.min = m
        self.max = n

    def set_subject(self, subject: pl.DataFrame) -> None:
        super().set_subject(subject)
        if self.min < 1:
            self.valid_starts = None

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        if self.min == 0:
            yield cursor
        stack = [(1, self.subpatterns[0]._op(ctxt, cursor))]
        while stack:
            try:
                i, cursor = stack[-1][0], next(stack[-1][1])
                if i >= self.min:
                    yield cursor
                if self.max is None or i < self.max:
                    stack.append((i + 1, self.subpatterns[0]._op(ctxt, cursor)))
            except StopIteration:
                stack.pop()


class ZeroOrMore(MToN):
    def __init__(self, pattern: Pattern) -> None:
        super().__init__(pattern, m=0)


class OneOrMore(MToN):
    def __init__(self, pattern: Pattern) -> None:
        super().__init__(pattern, m=1)


class OneOrZero(Pattern):
    def __init__(self, pattern: Pattern) -> None:
        super().__init__()
        self.subpatterns = [pattern]

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        yield from self.subpatterns[0]._op(ctxt, cursor)
        yield cursor


class Concat(Pattern):
    def __init__(self, pattern1: Pattern, pattern2: Pattern) -> None:
        super().__init__()
        self.subpatterns = [pattern1, pattern2]

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        for i in self.subpatterns[0]._op(ctxt, cursor):
            for j in self.subpatterns[1]._op(ctxt, i):
                yield j


class Alt(Pattern):
    def __init__(self, pattern1: Pattern, pattern2: Pattern) -> None:
        super().__init__()
        self.subpatterns = [pattern1, pattern2]

    def set_subject(self, subject: pl.DataFrame) -> None:
        super().set_subject(subject)
        if (self.subpatterns[0].valid_starts is not None) and (
            self.subpatterns[1].valid_starts is not None
        ):
            self.valid_starts = np.logical_or(
                self.subpatterns[0].valid_starts, self.subpatterns[1].valid_starts
            )
        else:
            self.valid_starts = None

    def _op(self, ctxt: ScanContext, cursor: int) -> Iterator[int]:
        yield from self.subpatterns[0]._op(ctxt, cursor)
        yield from self.subpatterns[1]._op(ctxt, cursor)


## compile token-level annotations into polars expressions


def pairwise_compose(fn: Callable[..., Pattern], items: list[Pattern]) -> Pattern:
    if len(items) == 1:
        return items[0]
    else:
        return fn(items[0], pairwise_compose(fn, items[1:]))


feature = pp.Word(pp.alphas + pp.nums)
number = pp.Word(pp.nums)
value = pp.QuotedString('"')

constraint_formula = pp.Forward()

atomic_constraint = (feature + "=" + value).set_parse_action(
    lambda toks: pl.col(toks[0]).str.contains("^" + toks[2] + "$")
)
constraint = atomic_constraint | pp.Suppress("(") + constraint_formula + pp.Suppress(
    ")"
)

token_conj = (
    constraint + pp.ZeroOrMore(pp.Suppress("&") + constraint)
).set_parse_action(lambda toks: toks[0].and_(*toks[1:]))
token_disj = (
    token_conj + pp.ZeroOrMore(pp.Suppress("|") + token_conj)
).set_parse_action(lambda toks: toks[0].or_(*toks[1:]))

constraint_formula <<= token_disj


node = (pp.Suppress("[") + constraint_formula + pp.Suppress("]")).set_parse_action(
    lambda x: Token(x)
) | (pp.Suppress("[") + pp.Empty() + pp.Suppress("]")).set_parse_action(
    lambda x: Skip()
)

## compile CQP commands into search operations

cqp = pp.Forward()

primary = node | pp.Suppress("(") + cqp + pp.Suppress(")")

repetition = (
    (primary + pp.Suppress("*")).set_parse_action(lambda e: ZeroOrMore(e[0]))
    | (primary + pp.Suppress("+")).set_parse_action(lambda e: OneOrMore(e[0]))
    | (primary + pp.Suppress("?")).set_parse_action(lambda e: OneOrZero(e[0]))
    | (
        primary
        + pp.Suppress("{")
        + number
        + pp.Suppress(",")
        + number
        + pp.Suppress("}")
    ).set_parse_action(lambda e: MToN(e[0], m=int(e[1]), n=int(e[2])))
    | (
        primary + pp.Suppress("{") + number + pp.Suppress(",") + pp.Suppress("}")
    ).set_parse_action(lambda e: MToN(e[0], m=int(e[1])))
    | (
        primary + pp.Suppress("{") + pp.Suppress(",") + number + pp.Suppress("}")
    ).set_parse_action(lambda e: MToN(e[0], n=int(e[1])))
    | (primary + pp.Suppress("{") + number + pp.Suppress("}")).set_parse_action(
        lambda e: MToN(e[0], m=int(e[1]), n=int(e[1]))
    )
    | primary
)

concatenation = (repetition + pp.ZeroOrMore(repetition)).set_parse_action(
    lambda toks: pairwise_compose(Concat, toks)
)

disjunction = concatenation + pp.ZeroOrMore(pp.Suppress("|") + concatenation)
disjunction = disjunction.set_parse_action(lambda toks: pairwise_compose(Alt, toks))

cqp <<= disjunction
