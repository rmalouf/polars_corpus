from __future__ import annotations

from typing import Any, Optional

import polars as pl
from lark import Lark, Transformer

from ._internal import Opcode

__all__ = ["cqp"]


GRAMMAR = r"""
cqp: disjunction

disjunction: concatenation ("|" concatenation)*

concatenation: repetition+

?repetition: primary
           | primary "*" -> rep_star
           | primary "+" -> rep_plus
           | primary "?" -> rep_question
           | primary "{" INT "}" -> rep_exact
           | primary "{" INT "," INT "}" -> rep_range_mn
           | primary "{" INT "," "}" -> rep_range_m
           | primary "{" "," INT "}" -> rep_range_n
           | primary "{" "," "}" -> rep_range_all

?primary: node
        | "(" cqp ")"
        | binding

binding: "$" NAME ":" (node | "(" cqp ")")

node: "[" constraint_formula? "]"

?constraint_formula: token_disj

token_disj: token_conj ("|" token_conj)*
token_conj: constraint ("&" constraint)*

?constraint: atomic
           | "(" constraint_formula ")"

atomic: NAME OP ESCAPED_STRING CASEI?

OP: "!=" | "="
CASEI: "%c"
NAME: /[a-zA-Z_][a-zA-Z_0-9]*/
INT: /[0-9]+/

%import common.ESCAPED_STRING
%import common.WS
%ignore WS
"""


def _star(body: list[Opcode]) -> list[Opcode]:
    opcodes: list[Opcode] = [Opcode.Split(1, len(body) + 2)]
    opcodes.extend(body)
    opcodes.append(Opcode.Jump(-(len(body) + 1)))
    return opcodes


def _question(body: list[Opcode]) -> list[Opcode]:
    opcodes: list[Opcode] = [Opcode.Split(1, len(body) + 1)]
    opcodes.extend(body)
    return opcodes


def _mn_expand(body: list[Opcode], m: int, n: Optional[int]) -> list[Opcode]:
    if n is not None and m > n:
        raise ValueError("m > n")
    opcodes: list[Opcode] = []
    for _ in range(m):
        opcodes.extend(body)
    if n is None:
        opcodes.extend(_star(body))
    else:
        for _ in range(n - m):
            opcodes.extend(_question(body))
    return opcodes


class _CQPCompiler(Transformer):
    def atomic(self, items: list[Any]) -> pl.Expr:
        feature = str(items[0])
        op = str(items[1])
        value = str(items[2])[1:-1]
        pattern = "^(" + value + ")$"
        if len(items) > 3:
            modifier = str(items[3])
            if modifier != "%c":
                raise ValueError(f"Unknown modifier: {modifier}")
            pattern = "(?i)" + pattern
        expr = pl.col(feature).str.contains(pattern)
        if op == "=":
            return expr
        return expr.not_()

    def token_conj(self, items: list[pl.Expr]) -> pl.Expr:
        if len(items) == 1:
            return items[0]
        return items[0].and_(*items[1:])

    def token_disj(self, items: list[pl.Expr]) -> pl.Expr:
        if len(items) == 1:
            return items[0]
        return items[0].or_(*items[1:])

    def node(self, items: list[pl.Expr]) -> list[Opcode]:
        if items:
            return [Opcode.Token(items[0].meta.serialize())]
        return [Opcode.Skip()]

    # See: Gimpel, James F. "A theory of discrete patterns and their implementation
    #      in SNOBOL4." Communications of the ACM 16, no. 2 (1973): 91-100.
    def binding(self, items: list[Any]) -> list[Opcode]:
        name = str(items[0])
        body: list[Opcode] = items[1]
        opcodes: list[Opcode] = [
            Opcode.PushVar(),
            Opcode.Split(3, 1),
            Opcode.PopVar(),
            Opcode.Fail(),
        ]
        opcodes.extend(body)
        opcodes.extend(
            [
                Opcode.BindVar(name),
                Opcode.Split(3, 1),
                Opcode.UnBindVar(),
                Opcode.Fail(),
            ]
        )
        return opcodes

    def rep_star(self, items: list[list[Opcode]]) -> list[Opcode]:
        return _star(list(items[0]))

    def rep_plus(self, items: list[list[Opcode]]) -> list[Opcode]:
        body = list(items[0])
        return body + [Opcode.Split(-len(body), 1)]

    def rep_question(self, items: list[list[Opcode]]) -> list[Opcode]:
        return _question(list(items[0]))

    def rep_exact(self, items: list[Any]) -> list[Opcode]:
        n = int(items[1])
        return _mn_expand(list(items[0]), n, n)

    def rep_range_mn(self, items: list[Any]) -> list[Opcode]:
        return _mn_expand(list(items[0]), int(items[1]), int(items[2]))

    def rep_range_m(self, items: list[Any]) -> list[Opcode]:
        return _mn_expand(list(items[0]), int(items[1]), None)

    def rep_range_n(self, items: list[Any]) -> list[Opcode]:
        return _mn_expand(list(items[0]), 0, int(items[1]))

    def rep_range_all(self, items: list[Any]) -> list[Opcode]:
        return _mn_expand(list(items[0]), 0, None)

    def concatenation(self, items: list[list[Opcode]]) -> list[Opcode]:
        opcodes: list[Opcode] = []
        for item in items:
            opcodes.extend(item)
        return opcodes

    def disjunction(self, items: list[list[Opcode]]) -> list[Opcode]:
        # Wrap each branch around the whole remaining alternation, so the jump
        # past a branch always skips every branch that follows it.
        opcodes: list[Opcode] = items[-1]
        for body in reversed(items[:-1]):
            opcodes = [
                Opcode.Split(1, len(body) + 2),
                *body,
                Opcode.Jump(len(opcodes) + 1),
                *opcodes,
            ]
        return opcodes

    def cqp(self, items: list[list[Opcode]]) -> list[Opcode]:
        return items[0]


_parser = Lark(GRAMMAR, start="cqp", parser="lalr", transformer=_CQPCompiler())


def cqp(query: str) -> list[Opcode]:
    """
    Compile a CQP query into the opcodes the matcher runs.

    Parameters
    ----------
    query : str
        CQP query, e.g. `[pos="NN.*"] [lemma="be"]`. See QUERY_LANGUAGE.md for
        the syntax.

    Returns
    -------
    list of Opcode
        The program the matcher runs, one opcode per step. A `Token` opcode
        carries the serialized Polars expression that tests a token, and
        `Split` and `Jump` encode alternation and repetition as relative
        offsets. The closing `Match` is not appended here; `search_cqp` adds
        it before running the program.

    Raises
    ------
    lark.exceptions.LarkError
        If `query` is not a well-formed CQP query.

    See Also
    --------
    polars_corpus.search_cqp : Run such a query against a corpus.
    """
    return _parser.parse(query)  # type: ignore[return-value]
