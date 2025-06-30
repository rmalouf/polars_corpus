from __future__ import annotations

from optparse import Option
from typing import Any, Optional

import polars as pl
import pyparsing as pp

from ._internal import Opcode, OpcodeMatcher, Span
from .search import SearchResults

__all__ = ["search", "Span"]


def col_name(i: int) -> str:
    return f"_{i}"


def compute_masks(df: pl.DataFrame, opcodes: list[Any]) -> pl.DataFrame:
    for pc in range(len(opcodes)):
        df = propagate_masks(pc, opcodes, df)
    df = df.select([col_name(i) for i in range(len(opcodes))])
    return df.rechunk()


def propagate_masks(pc: int, opcodes: list[Any], df: pl.DataFrame) -> pl.DataFrame:
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


def matchall(df: pl.DataFrame, query: str) -> Optional[SearchResults]:
    if df.is_empty():
        return None  # SearchResults(df, query, [])

    opcodes = list(cqp.parse_string(query, parse_all=True))
    opcodes.append((Opcode.MATCH,))

    mask_df = compute_masks(df, opcodes)
    masks = [mask_df.get_column(col) for col in mask_df.columns]
    opcode_matcher = OpcodeMatcher(opcodes, masks)

    return opcode_matcher.matchall()


def search(df: pl.DataFrame, query: str) -> SearchResults:
    return SearchResults(df, query, matchall(df, query))


## compile token-level constraints into polars expressions


feature = pp.Word(pp.alphas + pp.nums + "_")
number = pp.Word(pp.nums)
value = pp.QuotedString('"')

constraint_formula = pp.Forward()

atomic_constraint = (feature + "=" + value).set_parse_action(
    lambda toks: pl.col(toks[0]).str.contains("^(" + toks[2] + ")$")
)
constraint = atomic_constraint | pp.Suppress("(") + constraint_formula + pp.Suppress(
    ")"
)


def compile_atomic_constraint(args: pp.ParseResults) -> pl.Expr:
    expr = pl.col(args[0]).str.contains("^(" + args[2] + ")$")
    if args[1] == "=":
        return expr
    elif args[1] == "!=":
        return expr.not_()
    else:
        raise ValueError("Unknown constraint")


atomic_constraint = (
    feature + (pp.Literal("=") | pp.Literal("!=")) + value
).set_parse_action(compile_atomic_constraint)

constraint = atomic_constraint | pp.Suppress("(") + constraint_formula + pp.Suppress(
    ")"
)

token_conj = (
    constraint + pp.ZeroOrMore(pp.Suppress("&") + constraint)
).set_parse_action(lambda args: args[0].and_(*args[1:]))
token_disj = (
    token_conj + pp.ZeroOrMore(pp.Suppress("|") + token_conj)
).set_parse_action(lambda args: args[0].or_(*args[1:]))

constraint_formula <<= token_disj


def compile_node(args: pp.ParseResults) -> tuple[Opcode, ...]:
    if args:
        return (Opcode.TOKEN, args[0])
    else:
        return (Opcode.SKIP,)


node = (
    pp.Suppress("[") + (constraint_formula | pp.Empty()) + pp.Suppress("]")
).set_parse_action(compile_node)


## compile CQP commands into search operations

cqp = pp.Forward()

simple_primary = node | (pp.Suppress("(") + cqp + pp.Suppress(")"))
# binding = (feature + pp.Suppress(":") + simple_primary).set_parse_action(
#     lambda toks: Bind(toks[0], toks[1])
# )
# primary = simple_primary | binding
primary = simple_primary


def compile_star(args: pp.ParseResults) -> list[Any]:
    operations = [(Opcode.SPLIT, 1, len(args) + 2)]
    operations.extend(args)
    operations.append((Opcode.JUMP, -(len(args) + 1)))
    return operations


def compile_plus(args: pp.ParseResults) -> list[Any]:
    operations = args
    operations.append((Opcode.SPLIT, -len(args), 1))
    return operations


def compile_question(args: pp.ParseResults) -> list[Any]:
    operations = [(Opcode.SPLIT, 1, len(args) + 1)]
    operations.extend(args)
    return operations


def compile_m_to_n(args: pp.ParseResults):
    args_dict = args.as_dict()
    if "m_n" in args_dict:
        m = int(args_dict["m_n"])
        n = int(args_dict["m_n"])
    else:
        m = int(args_dict["m"]) if "m" in args_dict else 0
        n = int(args_dict["n"]) if "n" in args_dict else None
    if n and m > n:
        raise ValueError("m > n")

    operations = []
    for _ in range(m):
        operations.extend(args[0])
    if n is None:
        operations.extend(compile_star(args[0]))
    else:
        for i in range(0, n - m):
            operations.extend(compile_question(args[0]))
    return operations


repetition = (
    (primary + pp.Suppress("*")).set_parse_action(compile_star)
    | (primary + pp.Suppress("+")).set_parse_action(compile_plus)
    | (primary + pp.Suppress("?")).set_parse_action(compile_question)
    | (
        pp.Group(primary)
        + pp.Suppress("{")
        + pp.Opt(number).set_results_name("m")
        + pp.Suppress(",")
        + pp.Opt(number).set_results_name("n")
        + pp.Suppress("}")
    ).set_parse_action(compile_m_to_n)
    | (
        pp.Group(primary)
        + pp.Suppress("{")
        + number.set_results_name("m_n")
        + pp.Suppress("}")
    ).set_parse_action(compile_m_to_n)
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


def compile_disjunction(args: pp.ParseResults) -> list[Any]:
    if len(args) == 1:
        return args[0]
    else:
        operations = []
        for i in range(len(args) - 1):
            operations.append((Opcode.SPLIT, 1, len(args[i]) + 2))
            operations.extend(args[i])
            operations.append((Opcode.JUMP, len(args[i + 1]) + 1))
        operations.extend(args[-1])
        return operations


disjunction = (
    pp.Group(concatenation) + pp.ZeroOrMore(pp.Suppress("|") + pp.Group(concatenation))
).set_parse_action(compile_disjunction)

cqp <<= disjunction
