from __future__ import annotations

import time
from collections import deque, namedtuple, defaultdict
from enum import IntEnum
from typing import Iterator, Optional, Any

import numpy as np
import polars as pl
import pyparsing as pp
from tqdm import tqdm

from ._internal import Opcode, OpcodeMatcher

Span: type[Span] = namedtuple("Span", ("start", "end"))


type Mask = np.typing.NDArray[np.bool_]


class ScanContext:
    __slots__ = "bindings"

    # TODO: get rid of this class. pass everything as args so we can use local variables

    def __init__(self) -> None:
        self.bindings: dict[str, list[str]] = dict()


def compute_masks(df: pl.DataFrame, opcodes: list[Any]) -> Mask:
    token_exprs = []
    back_refs = defaultdict(list)
    columns = []
    for i, opcode in enumerate(opcodes):
        match opcode:
            case (Opcode.TOKEN, expr):
                token_exprs.append(expr.fill_null(False).alias(str(i)))
                columns.append(str(i))
                #print(f'Add {i}')
            case (Opcode.JUMP, offset):
                back_refs[i + offset].append(i)
            case (Opcode.SPLIT, offset1, offset2):
                back_refs[i + offset1].append(i)
                back_refs[i + offset2].append(i)
            case (Opcode.MATCH,) | (Opcode.SKIP,):
                token_exprs.append(pl.lit(True).alias(str(i)))
                columns.append(str(i))
                #print(f'Add {i}')
            case _:
                raise ValueError(f"Unknown opcode {opcode}")

    masks_df = df.select(token_exprs)

    agenda = [int(i) for i in columns]
    seen = set()
    while agenda:
        pc = agenda.pop(0)
        if pc not in seen:
            match opcodes[pc]:
                case (Opcode.JUMP, offset):
                    masks_df = masks_df.with_columns(pl.col(str(offset + pc)).alias(str(pc)))
                    #print(f'Add {pc}')
                    #masks[pc] = masks[offset + pc]
                case (Opcode.SPLIT, offset1, offset2):
                    masks_df = masks_df.with_columns((pl.col(str(offset1 + pc)) | pl.col(str(offset2 + pc))).alias(str(pc)))
                    #print(f'Add {pc}')
                    #masks[pc] = np.logical_or(masks[offset1 + pc], masks[offset2 + pc])
                case _:
                    pass
            agenda.extend(back_refs[pc])
            seen.add(pc)

    #masks_df = masks_df.collect()

    #masks = np.zeros((len(opcodes), len(df)), dtype=bool)
    #for col in masks_df.columns:
    #    masks[int(col), :] = masks_df[col].to_numpy()



    return masks_df.rechunk()


def matchall(
    df: pl.DataFrame, query: str, progress: bool = False
) -> Iterator[tuple[Span, dict[str, Any]]]:
    if progress:
        bar = tqdm(total=len(df))

    opcodes = list(cqp.parse_string(query, parse_all=True))
    opcodes.append((Opcode.MATCH,))

    now = time.time()
    mask = compute_masks(df, opcodes)
    print("Compute mask:", time.time() - now)
    now = time.time()

    print(mask.null_count())


    # opcodes = [(Opcode.TOKEN) if opcode[0]==Opcode.TOKEN else opcode  for opcode in opcodes]

    now = time.time()
    columns = [ ]
    masks = [ ]
    for col in sorted(mask.columns):
        columns.append(col)
        masks.append(mask[col])
    #    print(len(mask[col].get_chunks()))
    opcode_matcher = OpcodeMatcher(list(opcodes), masks)
    print("Init OpcodeMatcher:", time.time() - now)
    now = time.time()

    now = time.time()
    spans = opcode_matcher.matchall()
    print("Search:", time.time() - now)

    if progress:
        bar.update(len(df) - bar.n)
        bar.close()

    now = time.time()
    if spans is not None:
        spans = [Span(x, y) for x, y in spans]
    print("Result:", time.time() - now)

    return spans


def rust_match_opcodes(o, c):
    return o._match_opcodes(c)


def match_opcodes(opcodes: list[Any], masks: Mask, cursor: int) -> Optional[int]:
    match_end = cursor

    def _match(sp: int, pc: int) -> None:
        nonlocal match_end
        while True:
            if not masks[pc, sp]:
                break
            match opcodes[pc]:
                case (Opcode.TOKEN, *_) | (Opcode.SKIP, *_):
                    sp = sp + 1
                    pc = pc + 1
                case (Opcode.SPLIT, offset1, offset2):
                    _match(sp, pc + offset2)
                    pc = pc + offset1
                case (Opcode.JUMP, offset, *_):
                    pc = pc + offset
                case (Opcode.MATCH, *_):
                    match_end = max(match_end, sp)
                    break
                case _:
                    raise ValueError("Unknown opcode")

    _match(cursor, 0)
    if match_end > cursor:
        return match_end
    else:
        return None


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
    lambda x: (Opcode.TOKEN, x[0])
) | (pp.Suppress("[") + pp.Empty() + pp.Suppress("]")).set_parse_action(
    lambda x: (Opcode.SKIP,)
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
    result = [(Opcode.SPLIT, 1, len(args) + 2)]
    result.extend(args)
    result.append((Opcode.JUMP, -(len(args) + 1)))
    return result


def compile_plus(args):
    result = []
    result.extend(args)
    result.append((Opcode.SPLIT, -len(args), 1))
    return result


def compile_question(args):
    result = [(Opcode.SPLIT, 1, len(args) + 1)]
    result.extend(args)
    return result


def compile_m_to_n(args, m=None, n=None):
    result = []
    if m is None:
        m = 0
    else:
        for _ in range(m):
            result.extend(args)
    if n is None:
        result.extend(compile_star(args))
    else:
        for i in range(0, n - m):
            result.extend(compile_question(args))
    return result


repetition = (
    (primary + pp.Suppress("*")).set_parse_action(compile_star)
    | (primary + pp.Suppress("+")).set_parse_action(compile_plus)
    | (primary + pp.Suppress("?")).set_parse_action(compile_question)
    | (
        pp.Group(primary)
        + pp.Suppress("{")
        + number
        + pp.Suppress(",")
        + number
        + pp.Suppress("}")
    ).set_parse_action(lambda e: compile_m_to_n(e[0], m=int(e[1]), n=int(e[2])))
    | (
        pp.Group(primary)
        + pp.Suppress("{")
        + number
        + pp.Suppress(",")
        + pp.Suppress("}")
    ).set_parse_action(lambda e: compile_m_to_n(e[0], m=int(e[1])))
    | (
        pp.Group(primary)
        + pp.Suppress("{")
        + pp.Suppress(",")
        + number
        + pp.Suppress("}")
    ).set_parse_action(lambda e: compile_m_to_n(e[0], n=int(e[1])))
    | (
        pp.Group(primary) + pp.Suppress("{") + number + pp.Suppress("}")
    ).set_parse_action(lambda e: compile_m_to_n(e[0], m=int(e[1]), n=int(e[1])))
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
            result.append((Opcode.SPLIT, 1, len(args[i]) + 2))
            result.extend(args[i])
            result.append((Opcode.JUMP, len(args[i + 1]) + 1))
        result.extend(args[-1])
        return result


disjunction = (
    pp.Group(concatenation) + pp.ZeroOrMore(pp.Suppress("|") + pp.Group(concatenation))
).set_parse_action(compile_disjunction)

cqp <<= disjunction
