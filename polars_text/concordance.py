from __future__ import annotations

from typing import Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._typing import TPolarsFrame

__all__ = ["concordance"]


def concordance(
    df: TPolarsFrame,
    expr: pl.Expr,
    context: int = 5,
    left_context: Optional[int] = None,
    right_context: Optional[int] = None,
) -> TPolarsFrame:
    if left_context is not None or right_context is not None:
        if left_context is None:
            left_context = 0
        if right_context is None:
            right_context = 0
    else:
        left_context = context
        right_context = context

    if left_context < 0 or right_context < 0:
        raise ValueError("left_width and right_width must be non-negative")

    col_names = expr.meta.root_names()
    assert len(col_names) == 1, "expr must be a single column"
    col = pl.col(col_names[0])

    w = df.select(
        [col.shift(i).alias(f"context-{i}") for i in range(left_context, 0, -1)]
        + [col]
        + [col.shift(-i).alias(f"context+{i}") for i in range(1, right_context + 1)]
    ).filter(expr)

    return w
