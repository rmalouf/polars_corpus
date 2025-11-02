from __future__ import annotations

from pathlib import Path
from typing import cast

import polars as pl
from polars._typing import IntoExprColumn
from polars.plugins import register_plugin_function

LIB = Path(__file__).parent

__all__ = ["ttr", "msttr", "yules_k", "mtld"]


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
    expr_s: pl.Expr = cast(pl.Expr, pl.col(expr) if isinstance(expr, str) else expr)
    return expr_s.n_unique() / expr_s.len()


def msttr(expr: IntoExprColumn, n: int = 1000) -> pl.Expr:
    """
    Calculate mean segmental type-token ratio (MSTTR).

    MSTTR divides a text into consecutive non-overlapping segments of length n,
    calculates the TTR for each complete segment, and returns the mean of these
    values. Segments shorter than n are dropped.

    Parameters
    ----------
    expr : IntoExprColumn
        Column expression containing tokens to analyze.
    n : int, default=1000
        Segment size in tokens. Only complete segments of exactly this length
        are included in the calculation. Must be greater than 0.

    Returns
    -------
    pl.Expr
        Expression returning the mean segmental TTR as a float scalar.
        Returns null if there are no complete segments.

    Examples
    --------
    >>> # Create a corpus with 2500 tokens
    >>> tokens = ["word"] * 1000 + ["unique"] * 500 + ["word"] * 1000
    >>> df = pl.DataFrame({"tokens": tokens})
    >>> # Calculate MSTTR with default segment size (1000)
    >>> df.select(plc.msttr("tokens"))
    >>> # Calculate MSTTR with custom segment size
    >>> df.select(plc.msttr("tokens", n=500))

    Notes
    -----
    - If the text contains fewer than n tokens, the result is null

    Raises
    ------
    ValueError
        If n is not a positive integer.
    TypeError
        If n is not an integer.

    """
    if not isinstance(n, int):
        raise TypeError(f"Segment size n must be an integer, got {type(n).__name__}")
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


def mtld(expr: IntoExprColumn, threshold: float = 0.720) -> pl.Expr:
    """
    Calculate Measure of Textual Lexical Diversity (MTLD).

    MTLD (McCarthy & Jarvis, 2010) is a measure of lexical diversity that is
    relatively independent of text length. It works by counting "factors" -
    sequential stretches of text where the type-token ratio (TTR) remains above
    a threshold. The measure is calculated in both forward and backward
    directions and averaged.

    Parameters
    ----------
    expr : IntoExprColumn
        Column expression containing tokens to analyze.
    threshold : float, default=0.720
        TTR threshold for determining factor boundaries. The standard value
        is 0.720 (McCarthy & Jarvis, 2010). Must be strictly between 0 and 1.

    Returns
    -------
    pl.Expr
        Expression returning the MTLD score as a float scalar.
        Returns null if the text contains fewer than 10 tokens.
        Higher values indicate greater lexical diversity.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> # Text with low diversity (repeated words)
    >>> df = pl.DataFrame({"tokens": ["the"] * 10 + ["cat"] * 10})
    >>> df.select(plc.mtld("tokens"))
    >>> # Text with high diversity (unique words)
    >>> df = pl.DataFrame({"tokens": [f"word{i}" for i in range(100)]})
    >>> df.select(plc.mtld("tokens"))
    >>> # Use custom threshold
    >>> df.select(plc.mtld("tokens", threshold=0.800))

    Raises
    ------
    ValueError
        If threshold is not strictly between 0 and 1 (exclusive).

    References
    ----------
    McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation
    study of sophisticated approaches to lexical diversity assessment.
    Behavior Research Methods, 42(2), 381-392.

    """
    if not 0 < threshold < 1:
        raise ValueError(
            f"Threshold must be strictly between 0 and 1 (exclusive), got {threshold}"
        )

    expr_s = pl.col(expr) if isinstance(expr, str) else expr
    return register_plugin_function(
        plugin_path=LIB,
        args=[expr_s],
        function_name="py_mtld",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"threshold": threshold},
    )


def yules_k(expr: IntoExprColumn) -> pl.Expr:
    """
    Calculate Yule's K characteristic.

    Yule's K is a measure of lexical diversity that is relatively independent
    of text length. It is based on the frequency spectrum of word types,
    measuring how evenly vocabulary is distributed across frequency classes.
    Lower values indicate higher lexical diversity (more even distribution),
    while higher values indicate lower diversity (more repetition).

    Parameters
    ----------
    expr : IntoExprColumn
        Column expression containing tokens to analyze.

    Returns
    -------
    pl.Expr
        Expression returning Yule's K statistic as a float scalar.
        The value is scaled by 10,000 for readability.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> # Text with high diversity (low K)
    >>> df = pl.DataFrame({"tokens": [f"word{i}" for i in range(100)]})
    >>> df.select(plc.yules_k("tokens"))
    >>> # Text with low diversity (high K)
    >>> df = pl.DataFrame({"tokens": ["the"] * 50 + ["cat"] * 30 + ["sat"] * 20})
    >>> df.select(plc.yules_k("tokens"))

    References
    ----------
    Yule, G. U. (1944). The Statistical Study of Literary Vocabulary.
    Cambridge University Press.

    """
    expr_s: pl.Expr = cast(pl.Expr, pl.col(expr) if isinstance(expr, str) else expr)
    n = expr_s.len()
    spectrum = expr_s.unique_counts().alias("f").value_counts()
    inner = (spectrum.struct[0] ** 2 * spectrum.struct[1]) / (n * n)
    return 10000 * (inner.sum() - (1 / n))


## LEXICAL GROWTH CURVES
