from __future__ import annotations

from collections.abc import Iterable, Sequence
from difflib import get_close_matches
from typing import cast

import polars as pl
from polars._typing import IntoExprColumn

from ._typing import T_Frame

# The argument checks below are deliberately left out of `__all__`: they are for
# use inside the package, not part of its public API. Their job is to turn the
# mistakes users make -- a misspelled column, a string where a corpus belongs,
# an unknown method name -- into a message that says what to fix, before the
# error can resurface from inside a query plan where it is unreadable.
__all__ = ["ngrams", "output_name"]


def output_name(expr: IntoExprColumn) -> str:
    """Name of the column produced by `expr`.

    Parameters
    ----------
    expr : IntoExprColumn
        A column name, a Polars expression, or a Series.

    Returns
    -------
    str
        The name the column carries once `expr` is evaluated.
    """
    if isinstance(expr, str):
        return expr
    if isinstance(expr, pl.Series):
        return expr.name
    return expr.meta.output_name()


def as_corpus(frame: object, name: str = "corpus") -> pl.LazyFrame:
    """Check that `frame` is a usable corpus and return it lazily.

    Parameters
    ----------
    frame : object
        The value passed as a corpus.
    name : str, default "corpus"
        How to refer to it in error messages, e.g. "target corpus".

    Returns
    -------
    pl.LazyFrame
        `frame`, made lazy.

    Raises
    ------
    ValueError
        If `frame` is not a Polars DataFrame or LazyFrame, or is an empty
        DataFrame.
    """
    if not isinstance(frame, (pl.DataFrame, pl.LazyFrame)):
        raise ValueError(
            f"the {name} must be a polars DataFrame or LazyFrame, "
            f"got {type(frame).__name__}"
        )
    # A LazyFrame's height isn't known without running the query, so an empty
    # LazyFrame falls through to an empty result rather than this message.
    if isinstance(frame, pl.DataFrame) and frame.height == 0:
        raise ValueError(f"the {name} is empty")
    return frame.lazy()


def collect_like(result: pl.LazyFrame, source: T_Frame) -> T_Frame:
    """Give `result` back in the form `source` came in as.

    The other half of `as_corpus`: a function that takes either kind of frame,
    works lazily inside, and hands back a DataFrame to callers who passed one
    and a LazyFrame to callers who passed one.

    Parameters
    ----------
    result : pl.LazyFrame
        The query to return.
    source : DataFrame | LazyFrame
        The frame the caller passed in.

    Returns
    -------
    T_Frame
        `result` collected if `source` is a DataFrame, otherwise `result`.
    """
    # The eager/lazy correlation is real but not expressible: T_Frame is bound
    # by the argument types, while this branch is chosen at runtime.
    return cast(
        T_Frame, result.collect() if isinstance(source, pl.DataFrame) else result
    )


def check_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: Iterable[str],
    name: str = "corpus",
    param: str | None = None,
) -> None:
    """Check that `frame` has every column in `columns`.

    Parameters
    ----------
    frame : DataFrame | LazyFrame
        Corpus to check.
    columns : Iterable[str]
        Column names that must be present.
    name : str, default "corpus"
        How to refer to `frame` in error messages, e.g. "reference corpus".
    param : str, optional
        Keyword argument the caller took `columns` from, e.g. "file_id_column".
        Named in the error so the reader knows which argument to change.

    Raises
    ------
    ValueError
        If any of `columns` is missing, reporting the first one along with the
        columns the corpus does have.
    """
    have = frame.collect_schema().names()
    missing = [column for column in columns if column not in have]
    if missing:
        hint = f" Use {param}= to point at the right column." if param else ""
        raise ValueError(
            f"the {name} has no column {missing[0]!r}; "
            f"its columns are: {', '.join(have)}.{hint}"
        )


def as_expr(expr: object, param: str = "expr", hint: str = "") -> pl.Expr:
    """Turn a column name into an expression, or pass one through.

    Parameters
    ----------
    expr : object
        A column name or a Polars expression.
    param : str, default "expr"
        Name of the parameter being checked, used in error messages.
    hint : str, default ""
        Sentence appended to the error raised for a Series, explaining why the
        caller cannot accept one.

    Returns
    -------
    pl.Expr
        `expr` as an expression.

    Raises
    ------
    ValueError
        If `expr` is neither a column name nor an expression.
    """
    if isinstance(expr, str):
        return pl.col(expr)
    if isinstance(expr, pl.Expr):
        return expr
    if isinstance(expr, pl.Series):
        raise ValueError(
            f"{param} must be a column name or expression, not a Series.{hint} "
            f"Pass the column name instead, e.g. {expr.name!r}."
        )
    raise ValueError(
        f"{param} must be a column name or a polars expression, "
        f"got {type(expr).__name__}"
    )


def check_choice(value: object, options: Sequence[str], param: str = "method") -> str:
    """Match `value` against `options`, ignoring case and surrounding space.

    Parameters
    ----------
    value : object
        The value passed by the caller.
    options : Sequence[str]
        Accepted values, in lower case.
    param : str, default "method"
        Name of the parameter being checked, used in error messages.

    Returns
    -------
    str
        The matching entry of `options`.

    Raises
    ------
    ValueError
        If `value` is not a string or is not one of `options`. The message
        lists the options, and suggests one if `value` is close to it.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{param} must be one of {', '.join(options)}; got {type(value).__name__}"
        )
    canonical = value.strip().lower()
    if canonical not in options:
        close = get_close_matches(canonical, options, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"Unknown {param} {value!r}. Choose one of: {', '.join(options)}.{hint}"
        )
    return canonical


def ngrams(n: int, expr: pl.Expr | str) -> pl.Expr:
    if isinstance(expr, str):
        expr = pl.col(expr)
    exprs = [expr.alias("_0")] + [expr.shift(-i).alias(f"_{i}") for i in range(1, n)]
    return pl.struct(exprs)
