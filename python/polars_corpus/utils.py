from __future__ import annotations

import warnings
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

    Rejects non-frames and empty DataFrames. `name` is used in error messages.
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

    The other half of `as_corpus`: eager in, eager out.
    """
    # The eager/lazy correlation is real but not expressible: T_Frame is bound
    # by the argument types, while this branch is chosen at runtime.
    return cast(
        T_Frame, result.collect() if isinstance(source, pl.DataFrame) else result
    )


def drop_null_rows(
    lf: pl.LazyFrame, source: object, name: str = "corpus"
) -> pl.LazyFrame:
    """Drop the rows of `lf` holding a null in any column, and say how many went.

    Call this on a frame already cut down to the columns being read, so that
    nulls elsewhere in the corpus are none of its business. Counting the dropped
    rows means reading the frame, so the warning is raised only when `source` is
    eager -- the trade `as_corpus` makes for its empty-corpus check. `name` is
    used in the warning.
    """
    if isinstance(source, pl.DataFrame):
        columns = lf.collect_schema().names()
        total, dropped, *has_nulls = (
            lf.select(
                pl.len().alias("_total"),
                pl.any_horizontal(pl.all().is_null()).sum().alias("_dropped"),
                *(
                    pl.col(column).is_null().any().alias(f"_null_{i}")
                    for i, column in enumerate(columns)
                ),
            )
            .collect()
            .row(0)
        )
        if dropped:
            culprits = [c for c, seen in zip(columns, has_nulls) if seen]
            warnings.warn(
                f"dropped {dropped:,} of {total:,} rows of the {name} holding a "
                f"null {' or '.join(repr(c) for c in culprits)}",
                # Report against the caller of the function that called us.
                stacklevel=3,
            )
    return lf.drop_nulls()


def check_columns(
    frame: pl.DataFrame | pl.LazyFrame,
    columns: Iterable[str],
    name: str = "corpus",
    param: str | None = None,
) -> None:
    """Check that `frame` has every column in `columns`.

    Reports the first one missing along with the columns `frame` does have.
    `name` and `param` are used in error messages, `param` naming the keyword
    argument the reader should change.
    """
    have = frame.collect_schema().names()
    missing = [column for column in columns if column not in have]
    if missing:
        hint = f" Use {param}= to point at the right column." if param else ""
        raise ValueError(
            f"the {name} has no column {missing[0]!r}; "
            f"its columns are: {', '.join(have)}.{hint}"
        )


def check_expr(
    frame: pl.DataFrame | pl.LazyFrame,
    expr: pl.Expr,
    name: str = "corpus",
    param: str = "expr",
) -> str:
    """Check that `frame` can evaluate `expr`, and name the column it produces.

    Resolving against the schema reads no data and settles what a column name
    alone cannot: regexes, selectors and positional references name their
    columns indirectly. `name` and `param` are used in error messages.
    """
    try:
        resolved = frame.lazy().select(expr).collect_schema().names()
    except pl.exceptions.ColumnNotFoundError as err:
        # An expression that names its columns plainly can say which is missing.
        check_columns(frame, expr.meta.root_names(), name)
        raise ValueError(f"the {name} cannot evaluate {param}: {err}") from err
    if len(resolved) != 1:
        raise ValueError(
            f"{param} must identify a single column, but against the {name} it "
            f"selects {', '.join(resolved) or 'none'}"
        )
    return resolved[0]


def as_expr(expr: object, param: str = "expr", hint: str = "") -> pl.Expr:
    """Turn a column name into an expression, or pass one through.

    `param` is used in error messages; `hint` is appended to the one raised for
    a Series, to say why the caller cannot accept one.
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

    `options` are in lower case. A bad value gets an error listing them, with a
    difflib suggestion if it is close to one. `param` names it in the message.
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
