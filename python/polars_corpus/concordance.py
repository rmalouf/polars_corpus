from __future__ import annotations

import polars as pl

from .cqp_parser import cqp

__all__ = ["Concordance"]


class Concordance:
    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        longest_match: bool = True,
        progress: bool = False,
    ):
        self.df = df
        self.query = query
        self.spans = list(
            cqp.parse_string(query, parse_all=True)[0].get_matches(
                df, longest_match=longest_match, progress=progress
            )
        )

    # def show_kwic(self):


#
#
# def concordance(df: TPolarsFrame, expr: pl.Expr, context: int = 5) -> TPolarsFrame:
#     left_context = context
#     right_context = context
#     if left_context < 0 or right_context < 0:
#         raise ValueError("left_width and right_width must be non-negative")
#
#     col_names = expr.meta.root_names()
#     assert len(col_names) == 1, "expr must be a single column"
#     col = pl.col(col_names[0])
#
#     w = df.select(
#         [col.shift(i).alias(f"context-{i}") for i in range(left_context, 0, -1)]
#         + [col]
#         + [col.shift(-i).alias(f"context+{i}") for i in range(1, right_context + 1)]
#     ).filter(expr)
#
#     return w
