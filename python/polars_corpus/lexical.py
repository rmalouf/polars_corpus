from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.plugins import register_plugin_function

from ._typing import IntoExpr
from .utils import as_expr

LIB = Path(__file__).parent

__all__ = ["ttr", "msttr", "yules_k", "mtld"]


def ttr(expr: IntoExpr) -> pl.Expr:
    """
    Calculate type-token ratio (TTR).

    The type-token ratio is the ratio of unique tokens (types) to total tokens.
    TTR ranges from 0 to 1, with higher values indicating greater lexical diversity.

    Parameters
    ----------
    expr : IntoExpr
        Column name or expression holding the tokens to analyze.

    Returns
    -------
    pl.Expr
        Expression returning the type-token ratio as a float scalar.
    """
    tokens = as_expr(expr)
    return tokens.n_unique() / tokens.len()


def msttr(expr: IntoExpr, n: int = 1000) -> pl.Expr:
    """
    Calculate mean segmental type-token ratio (MSTTR).

    MSTTR divides a text into consecutive non-overlapping segments of length n,
    calculates the TTR for each complete segment, and returns the mean of these
    values. Segments shorter than n tokens are dropped.

    Parameters
    ----------
    expr : IntoExpr
        Column name or expression holding the tokens to analyze.
    n : int, default=1000
        Segment size in tokens. Only complete segments of exactly this length
        are included in the calculation. Must be greater than 0.

    Returns
    -------
    pl.Expr
        Expression returning the mean segmental TTR as a float scalar.
        Returns null if the text contains fewer than n tokens.

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

    return register_plugin_function(
        plugin_path=LIB,
        args=[as_expr(expr)],
        function_name="py_msttr",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"n": n},
    )


def mtld(expr: IntoExpr, threshold: float = 0.720) -> pl.Expr:
    """
    Calculate Measure of Textual Lexical Diversity (MTLD).

    MTLD (McCarthy & Jarvis, 2010) is a measure of lexical diversity that is
    relatively independent of text length. It works by counting "factors", i.e.,
    sequential stretches of text where the type-token ratio (TTR) remains above
    a threshold. The measure is calculated in both forward and backward
    directions and the two values are averaged.

    Parameters
    ----------
    expr : IntoExpr
        Column name or expression holding the tokens to analyze.
    threshold : float, default=0.720
        TTR threshold for determining factor boundaries. The standard value
        is 0.720 (McCarthy & Jarvis, 2010). Must be strictly between 0 and 1.

    Returns
    -------
    pl.Expr
        Expression returning the MTLD score as a float scalar.
        Returns null if the text contains fewer than 10 tokens.
        Higher values indicate greater lexical diversity.

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

    return register_plugin_function(
        plugin_path=LIB,
        args=[as_expr(expr)],
        function_name="py_mtld",
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"threshold": threshold},
    )


def yules_k(expr: IntoExpr) -> pl.Expr:
    """
    Calculate Yule's K characteristic.

    Yule's K is a measure of lexical diversity that is relatively independent
    of text length. It is based on the frequency spectrum of word types,
    measuring how evenly vocabulary is distributed across frequency classes.
    Lower values indicate higher lexical diversity (more even distribution),
    while higher values indicate lower diversity (more repetition).

    Parameters
    ----------
    expr : IntoExpr
        Column name or expression holding the tokens to analyze.

    Returns
    -------
    pl.Expr
        Expression returning Yule's K statistic as a float scalar.
        The value is scaled by 10,000 for readability.

    References
    ----------
    Yule, G. U. (1944). The Statistical Study of Literary Vocabulary.
    Cambridge University Press.
    """
    tokens = as_expr(expr)
    n = tokens.len()
    spectrum = tokens.unique_counts().alias("f").value_counts()
    inner = (spectrum.struct[0] ** 2 * spectrum.struct[1]) / (n * n)
    return 10000 * (inner.sum() - (1 / n))


## LEXICAL GROWTH CURVES
