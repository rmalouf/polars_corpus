from __future__ import annotations

from pathlib import Path

import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

__all__ = ["ttr", "msttr"]


def ttr(expr: IntoExprColumn) -> pl.Expr:
    """
    Calculate type-token ratio (TTR).

    The type-token ratio is the ratio of unique tokens (types) to total tokens.
    TTR ranges from 0 to 1, with higher values indicating greater lexical diversity.

    Parameters
    ----------
    expr : IntoExprColumn
        Column expression containing tokens to analyze.

    Returns
    -------
    pl.Expr
        Expression returning the type-token ratio as a float.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> df = pl.DataFrame({"words": ["the", "cat", "sat", "on", "the", "mat"]})
    >>> df.select(plc.ttr("words"))
    shape: (1, 1)
    ┌───────┐
    │ words │
    │ ---   │
    │ f64   │
    ╞═══════╡
    │ 0.833 │
    └───────┘

    """
    expr_s = pl.col(expr) if isinstance(expr, str) else expr
    n_unique = expr_s.n_unique()
    return (n_unique if isinstance(n_unique, pl.Expr) else pl.lit(n_unique)).cast(
        pl.Float64
    ) / expr_s.len()


def msttr(expr: IntoExprColumn, n: int = 1000) -> pl.Expr:
    """
    Calculate mean segmental type-token ratio (MSTTR).

    MSTTR divides a text into consecutive non-overlapping segments of length n,
    calculates the TTR for each complete segment, and returns the mean of these
    values. Segments shorter than n are dropped. This metric is more stable than
    simple TTR for texts of varying lengths.

    Parameters
    ----------
    expr : IntoExprColumn
        Column expression containing tokens to analyze.
    n : int, default=1000
        Segment size in tokens. Only complete segments of exactly this length
        are included in the calculation.

    Returns
    -------
    pl.Expr
        Expression returning the mean segmental TTR as a float scalar.
        Returns null if there are no complete segments.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> # Create a corpus with 2500 tokens
    >>> tokens = ["word"] * 1000 + ["unique"] * 500 + ["word"] * 1000
    >>> df = pl.DataFrame({"tokens": tokens})
    >>> # Calculate MSTTR with default segment size (1000)
    >>> df.select(plc.msttr("tokens"))
    >>> # Calculate MSTTR with custom segment size
    >>> df.select(plc.msttr("tokens", n=500))

    Notes
    -----
    - The final incomplete segment is always dropped
    - If the text contains fewer than n tokens, the result is null
    - All data types are converted to strings for comparison

    """
    if n <= 0:
        raise ValueError(f"Segment size n must be greater than 0, got {n}")

    expr_s = pl.col(expr) if isinstance(expr, str) else expr
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr_s],
        function_name="py_msttr",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"n": n},
    )


## LEXICAL GROWTH CURVES
