from __future__ import annotations

import time
from typing import Any, Iterator, Optional

import polars as pl
import pyparsing as pp

from ._internal import Opcode, OpcodeMatcher, Span


def col_name(i):
    return f"_{i}"


def compute_masks(df: pl.DataFrame, opcodes: list[Any]) -> pl.DataFrame:
    for pc in range(len(opcodes)):
        df = propagate_masks(pc, opcodes, df)
    df = df.select([col_name(i) for i in range(len(opcodes))])
    return df.rechunk()


def propagate_masks(pc, opcodes, df) -> pl.DataFrame:
    if col_name(pc) not in df:
        match opcodes[pc]:
            case (Opcode.TOKEN, expr):
                df = df.with_columns(expr.fill_null(False).alias(col_name(pc)))
            case (Opcode.MATCH,) | (Opcode.SKIP,):
                df = df.with_columns(pl.lit(True).alias(col_name(pc)))
            case (Opcode.JUMP, offset):
                if col_name(pc + offset) not in df.columns:
                    df = propagate_masks(pc + offset, opcodes, df)
                df = df.with_columns(pl.col(col_name(pc + offset)).alias(col_name(pc)))
            case (Opcode.SPLIT, offset1, offset2):
                if col_name(pc + offset1) not in df.columns:
                    df = propagate_masks(pc + offset1, opcodes, df)
                if col_name(pc + offset2) not in df.columns:
                    df = propagate_masks(pc + offset2, opcodes, df)
                df = df.with_columns(
                    (
                        pl.col(col_name(pc + offset1)) | pl.col(col_name(pc + offset2))
                    ).alias(col_name(pc))
                )
            case _:
                raise ValueError(f"Unknown opcode {opcodes[pc]}")
    return df


def matchall(df: pl.DataFrame, query: str) -> Iterator[tuple[Span, dict[str, Any]]]:
    now = time.time()

    opcodes = list(cqp.parse_string(query, parse_all=True))
    opcodes.append((Opcode.MATCH,))

    mask = compute_masks(df, opcodes)
    #    print("Compute mask:", time.time() - now)
    #    now = time.time()

    #    now = time.time()
    columns = []
    masks = []
    for col in sorted(mask.columns):
        columns.append(col)
        masks.append(mask[col])
    #    print(len(mask[col].get_chunks()))
    opcode_matcher = OpcodeMatcher(list(opcodes), masks)
    #    print("Init OpcodeMatcher:", time.time() - now)
    #    now = time.time()

    #    now = time.time()
    spans = opcode_matcher.matchall()
    #    print("Search:", time.time() - now)

    print(f"{time.time()-now:.3f} seconds")

    return spans


## compile token-level constraints into polars expressions


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
