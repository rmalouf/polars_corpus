from __future__ import annotations

from collections import deque, namedtuple, defaultdict
from typing import Iterator, Optional, Any

import numpy as np
import polars as pl
import pyparsing as pp
from tqdm import tqdm

Token: type[Token] = namedtuple("Token", ("expr"))
Jump: type[Jump] = namedtuple("Jump", ("offset"))
Split: type[Split] = namedtuple("Split", ("offset1", "offset2"))
Match: type[Match] = namedtuple("Match", ())

Span: type[Span] = namedtuple("Span", ("start", "end"))


type Mask = np.typing.NDArray[np.bool_]

class ScanContext:
    __slots__ = "bindings"

    # TODO: get rid of this class. pass everything as args so we can use local variables

    def __init__(self) -> None:
        self.bindings: dict[str, list[str]] = dict()


def compute_masks(df: pl.DataFrame, opcodes: list[Any]) -> Mask:
    print("1")
    token_exprs = []
    back_refs = defaultdict(list)
    for i, opcode in enumerate(opcodes):
        match opcode:
            case Token(expr):
                token_exprs.append(opcode.expr.alias(str(i)))
            case Jump(offset):
                back_refs[i + offset].append(i)
            case Split(offset1, offset2):
                back_refs[i + offset1].append(i)
                back_refs[i + offset2].append(i)
            case Match():
                token_exprs.append(pl.lit(True).alias(str(i)))
            case _:
                pass

    print("2")
    masks_df = df.select(token_exprs)
    print("3")

    masks = np.zeros((len(opcodes), len(df)), dtype=bool)
    for col in masks_df.columns:
        masks[int(col), :] = masks_df[col].to_numpy()

    print("4")
    agenda = [int(i) for i in masks_df.columns]
    seen = set()
    while agenda:
        pc = agenda.pop(0)
        if pc not in seen:
            match opcodes[pc]:
                case Jump(offset):
                    masks[pc] = masks[offset + pc]
                case Split(offset1, offset2):
                    masks[pc] = np.logical_or(masks[offset1 + pc], masks[offset2 + pc])
                case _:
                    pass
            agenda.extend(back_refs[pc])
            seen.add(pc)

    return masks


def matchall(
    df: pl.DataFrame, query: str, progress: bool = False
) -> Iterator[tuple[Span, dict[str, Any]]]:
    if progress:
        bar = tqdm(total=len(df))

    opcodes = list(cqp.parse_string(query, parse_all=True))
    opcodes.append(Match())

    masks = compute_masks(df, opcodes)

    starts = np.where(masks[0, :])[0]
    n = len(starts)
    i = 0
    while i < n:
        cursor = starts[i]
        if progress:
            bar.update(cursor - bar.n)
        if (match_end := match_opcodes(opcodes, masks, cursor)) is None:
            i = i + 1
        else:
            yield Span(cursor, match_end), {}
            while i < n and starts[i] < match_end:
                i = i + 1

    if progress:
        bar.update(len(df) - bar.n)
        bar.close()


def match_opcodes(
    opcodes: list[Any], masks: Mask, cursor: int, pc: int = 0
) -> int:
    while True:
        if not masks[pc, cursor]:
            break
        match opcodes[pc]:
            case Token(_):
                cursor = cursor + 1
                pc = pc + 1
            case Split(offset1, offset2):
                if (
                    match_end := match_opcodes(opcodes, masks, cursor, pc + offset1)
                ) is None:
                    pc = pc + offset2
                else:
                    return match_end
            case Jump(offset):
                pc = pc + offset
            case Match():
                return cursor


## compile token-level annotations into polars expressions


feature = pp.Word(pp.alphas + pp.nums)
number = pp.Word(pp.nums)
value = pp.QuotedString('"')

constraint_formula = pp.Forward()

atomic_constraint = (feature + "=" + value).set_parse_action(
    lambda toks: pl.col(toks[0]).str.contains("^(" + toks[2] + ")$")
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
    lambda x: Token(x[0])
) | (pp.Suppress("[") + pp.Empty() + pp.Suppress("]")).set_parse_action(
    lambda x: Skip()
)

## compile CQP commands into search operations

cqp = pp.Forward()

# TODO: bindings don't work right with multi-token matches

simple_primary = node | (pp.Suppress("(") + cqp + pp.Suppress(")"))
# binding = (feature + pp.Suppress(":") + simple_primary).set_parse_action(
#     lambda toks: Bind(toks[0], toks[1])
# )
# primary = simple_primary | binding
primary = simple_primary


def compile_star(args):
    result = [Split(1, len(args) + 2)]
    result.extend(args)
    result.append(Jump(-(len(args) + 1)))
    return result


def compile_plus(args):
    result = []
    result.extend(args)
    result.append(Split(-len(args), 1))
    return result


def compile_question(args):
    result = [Split(1, len(args) + 1)]
    result.extend(args)
    return result


#
# def compile_m_to_n(args, m, n):
#     result = [ ]
#     if m == 0:
#         result.extend(compile_question(compile_m_to_n(args, 1, n)))
#     else:
#         for _ in range(m):
#             result.append(args)


repetition = (
    (primary + pp.Suppress("*")).set_parse_action(compile_star)
    | (primary + pp.Suppress("+")).set_parse_action(compile_plus)
    | (primary + pp.Suppress("?")).set_parse_action(compile_question)
    | primary
)


# repetition = (
#     (primary + pp.Suppress("*")).set_parse_action(compile_star)
#     | (primary + pp.Suppress("+")).set_parse_action(lambda e: OneOrMore(e[0]))
#     | (primary + pp.Suppress("?")).set_parse_action(lambda e: OneOrZero(e[0]))
#     | (
#         primary
#         + pp.Suppress("{")
#         + number
#         + pp.Suppress(",")
#         + number
#         + pp.Suppress("}")
#     ).set_parse_action(lambda e: MToN(e[0], m=int(e[1]), n=int(e[2])))
#     | (
#         primary + pp.Suppress("{") + number + pp.Suppress(",") + pp.Suppress("}")
#     ).set_parse_action(lambda e: MToN(e[0], m=int(e[1])))
#     | (
#         primary + pp.Suppress("{") + pp.Suppress(",") + number + pp.Suppress("}")
#     ).set_parse_action(lambda e: MToN(e[0], n=int(e[1])))
#     | (primary + pp.Suppress("{") + number + pp.Suppress("}")).set_parse_action(
#         lambda e: MToN(e[0], m=int(e[1]), n=int(e[1]))
#     )
#     | primary
# )

concatenation = pp.OneOrMore(repetition)


def compile_disjunction(args):
    if len(args) == 1:
        return args[0]
    else:
        result = []
        for i in range(len(args) - 1):
            result.append(Split(1, len(args[i]) + 2))
            result.extend(args[i])
            result.append(Jump(len(args[i + 1]) + 1))
        result.extend(args[-1])
        return result


disjunction = (
    pp.Group(concatenation) + pp.ZeroOrMore(pp.Suppress("|") + pp.Group(concatenation))
).set_parse_action(compile_disjunction)

cqp <<= disjunction
