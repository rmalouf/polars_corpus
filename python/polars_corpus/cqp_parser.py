from __future__ import annotations

from typing import Any, Optional

import polars as pl
import pyparsing as pp

from ._internal import Opcode

__all__ = ["cqp"]


feature = pp.Word(pp.alphas + pp.nums + "_")
number = pp.Word(pp.nums)
value = pp.QuotedString('"')
case_modifier = pp.Optional(pp.Literal("%c"))
variable = pp.Suppress("$") + pp.Word(pp.alphas + pp.nums + "_")

constraint_formula = pp.Forward()


def compile_atomic_constraint(args: pp.ParseResults) -> pl.Expr:
    pattern = "^(" + args[2] + ")$"
    case_insensitive = len(args) > 3 and args[3] == "%c"
    if case_insensitive:
        pattern = "(?i)" + pattern

    expr = pl.col(args[0]).str.contains(pattern)
    if args[1] == "=":
        return expr
    elif args[1] == "!=":
        return expr.not_()
    else:
        raise ValueError("Unknown constraint")


atomic_constraint = (
    feature + (pp.Literal("=") | pp.Literal("!=")) + value + case_modifier
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


def compile_node(args: pp.ParseResults) -> Opcode:
    if args:
        return Opcode.Token(args[0].meta.serialize())
    else:
        return Opcode.Skip()


node = (
    pp.Suppress("[") + (constraint_formula | pp.Empty()) + pp.Suppress("]")
).set_parse_action(compile_node)


## compile CQP commands into search Opcodes

cqp = pp.Forward()


## See: Gimpel, James F. "A theory of discrete patterns and their implementation
##      in SNOBOL4." Communications of the ACM 16, no. 2 (1973): 91-100.


def compile_binding(args: pp.ParseResults) -> list[Opcode]:
    opcodes: list[Opcode] = []
    opcodes.append(Opcode.PushVar())
    opcodes.append(Opcode.Split(3, 1))
    opcodes.append(Opcode.PopVar())
    opcodes.append(Opcode.Fail())
    opcodes.extend(args[1:])
    opcodes.append(Opcode.BindVar(args[0]))
    opcodes.append(Opcode.Split(3, 1))
    opcodes.append(Opcode.UnBindVar())
    opcodes.append(Opcode.Fail())
    return opcodes


simple_primary = node | (pp.Suppress("(") + cqp + pp.Suppress(")"))
binding = (
    variable + pp.Suppress(":") + pp.Suppress("(") + cqp + pp.Suppress(")")
).set_parse_action(compile_binding)
primary = simple_primary | binding


def compile_star(args: pp.ParseResults) -> list[Opcode]:
    opcodes: list[Opcode] = [Opcode.Split(1, len(args) + 2)]

    opcodes.extend(args)
    opcodes.append(Opcode.Jump(-(len(args) + 1)))
    return opcodes


def compile_plus(args: pp.ParseResults) -> list[Opcode]:
    opcodes: list[Opcode] = []
    opcodes.extend(args)
    opcodes.append(Opcode.Split(-len(args), 1))
    return opcodes


def compile_question(args: pp.ParseResults) -> list[Opcode]:
    opcodes: list[Opcode] = [Opcode.Split(1, len(args) + 1)]
    opcodes.extend(args)
    return opcodes


def compile_m_to_n(args: pp.ParseResults) -> list[Opcode]:
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

    opcodes: list[Opcode] = []
    for _ in range(m):
        opcodes.extend(args[0])
    if n is None:
        opcodes.extend(compile_star(args[0]))
    else:
        for i in range(0, n - m):
            opcodes.extend(compile_question(args[0]))
    return opcodes


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
        opcodes: list[Opcode] = []
        for i in range(len(args) - 1):
            opcodes.append(Opcode.Split(1, len(args[i]) + 2))
            opcodes.extend(args[i])
            opcodes.append(Opcode.Jump(len(args[i + 1]) + 1))
        opcodes.extend(args[-1])
        return opcodes


disjunction = (
    pp.Group(concatenation) + pp.ZeroOrMore(pp.Suppress("|") + pp.Group(concatenation))
).set_parse_action(compile_disjunction)

cqp <<= disjunction
