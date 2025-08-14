from __future__ import annotations

from typing import Any, Optional

import polars as pl
import pyparsing as pp

from ._internal import Opcode

__all__ = ["cqp"]


## compile token-level constraints into polars expressions


feature = pp.Word(pp.alphas + pp.nums + "_")
number = pp.Word(pp.nums)
value = pp.QuotedString('"')

constraint_formula = pp.Forward()


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
    operations: list[Any] = [(Opcode.SPLIT, 1, len(args) + 2)]
    operations.extend(args)
    operations.append((Opcode.JUMP, -(len(args) + 1)))
    return operations


def compile_plus(args: pp.ParseResults) -> list[Any]:
    operations: list[Any] = []
    operations.extend(args)
    operations.append((Opcode.SPLIT, -len(args), 1))
    return operations


def compile_question(args: pp.ParseResults) -> list[Any]:
    operations: list[Any] = [(Opcode.SPLIT, 1, len(args) + 1)]
    operations.extend(args)
    return operations


def compile_m_to_n(args: pp.ParseResults) -> list[Any]:
    m: Optional[int]
    n: Optional[int]
    args_dict = args.as_dict()
    if "m_n" in args_dict:
        m = int(args_dict["m_n"])
        n = int(args_dict["m_n"])
    else:
        m = int(args_dict["m"]) if "m" in args_dict else 0
        n = int(args_dict["n"]) if "n" in args_dict else None
    if n and m > n:
        raise ValueError("m > n")

    operations: list[Any] = []
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


concatenation = pp.OneOrMore(repetition)


def compile_disjunction(args: pp.ParseResults) -> Any:
    if len(args) == 1:
        return args[0]
    else:
        operations: list[Any] = []
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