from __future__ import annotations

from collections.abc import Iterable, Sequence
from difflib import get_close_matches
from typing import cast, Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._typing import IntoExpr, Measure, T_Frame


__all__ = ["ngrams", "proportion", "is_letters"]


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


def as_eager(frame: object, name: str = "corpus") -> pl.DataFrame:
    """Check that `frame` is a corpus a lazy one cannot stand in for.

    For work that has to see the rows themselves: searching walks the corpus by
    position and hands its columns to Rust, neither of which a query plan can
    offer. An empty corpus passes -- it has no rows to match, not no rows to
    read. `name` is used in error messages.
    """
    if not isinstance(frame, pl.DataFrame):
        hint = (
            " Call .collect() on it first." if isinstance(frame, pl.LazyFrame) else ""
        )
        raise ValueError(
            f"the {name} must be an eager polars DataFrame, "
            f"got {type(frame).__name__}.{hint}"
        )
    return frame


def collect_like(result: pl.LazyFrame, source: T_Frame) -> T_Frame:
    """Give `result` back in the form `source` came in as.

    The other half of `as_corpus`: eager in, eager out.
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
    param: Optional[str] = None,
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


def check_choices(
    value: object, options: Sequence[str], param: str = "method"
) -> list[str]:
    """Match `value`, one option or a list of them, against `options`.

    Each is normalized as `check_choice` normalizes it. Repeats are dropped,
    keeping the order asked for, so callers can name the columns they produce
    from the result. `param` names the argument in error messages.
    """
    if isinstance(value, str):
        return [check_choice(value, options, param)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{param} must be one of {', '.join(options)}, or a list of them; "
            f"got {type(value).__name__}"
        )
    if not value:
        raise ValueError(
            f"{param} is empty; name at least one of: {', '.join(options)}"
        )
    return list(dict.fromkeys(check_choice(item, options, param) for item in value))


def check_measure(
    value: object, options: Sequence[str], param: str = "method"
) -> str | Measure:
    """Match `value` against `options`, or pass a user-written measure through.

    A callable is taken as a measure of the caller's own and returned unchanged;
    anything else is normalized as `check_choice` normalizes it.
    """
    if callable(value):
        return cast(Measure, value)
    return check_choice(value, options, param)


def check_measures(
    value: object, options: Sequence[str], param: str = "method"
) -> list[str | Measure]:
    """Match `value`, one measure or a list of them, against `options`.

    As `check_choices`, but a callable item passes through as a measure of the
    caller's own. Repeats are dropped, keeping the order asked for.
    """
    if isinstance(value, str) or callable(value):
        return [check_measure(value, options, param)]
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"{param} must be one of {', '.join(options)}, a function, or a "
            f"list of them; got {type(value).__name__}"
        )
    if not value:
        raise ValueError(
            f"{param} is empty; name at least one of: {', '.join(options)}"
        )
    # remove duplicate methods
    return list(dict.fromkeys(check_measure(item, options, param) for item in value))


def ngrams(n: int, in_expr: IntoExpr, as_str: bool = True) -> pl.Expr:
    """Gather each token together with the `n - 1` tokens that follow it.

    Parameters
    ----------
    n : int
        Length of the n-gram: 1 for unigrams, 2 for bigrams, and so on.
    in_expr : IntoExpr
        Column name or expression holding the tokens to gather.
    as_str : bool, default True
        Join the tokens into one space-separated string. Set to False for a
        list column instead.

    Returns
    -------
    pl.Expr
        Expression giving the n-gram starting at each row. The last `n - 1`
        rows have no full n-gram to start and come out null, as does any
        n-gram covering a null token.

    Notes
    -----
    Rows are read in the order the frame holds them, so an n-gram will run
    from the end of one file into the start of the next. Add `.over()` on the
    file id column to stop them at the boundary:

    ```>>> df.select(pl.col("token").corpus.ngrams(2).over("file_id"))```
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    expr = as_expr(in_expr)
    exprs = [expr.shift(-i) for i in range(n)]

    if as_str:
        # concat_str is null for the whole n-gram if any token in it is null.
        return pl.concat_str(exprs, separator=" ")
    # concat_list is not: a null token would leave a shorter n-gram behind
    # looking like a real one, so null those rows out to match.
    return (
        pl.when(pl.any_horizontal(sub.is_null() for sub in exprs))
        .then(None)
        .otherwise(pl.concat_list(exprs))
    )


def proportion(in_expr: IntoExpr, in_group_by: Optional[IntoExpr] = None) -> pl.Expr:
    """Rescale counts as a share of their total.

    Parameters
    ----------
    in_expr : IntoExpr
        Column name or expression holding the counts.
    in_group_by : IntoExpr, optional
        Column name or expression to take the total within. By default the
        total is over the whole column.

    Returns
    -------
    pl.Expr
        Expression giving each count divided by the total, so the values sum to 1.
    """
    expr = as_expr(in_expr)
    if in_group_by is None:
        return expr / expr.sum()
    else:
        group_by = as_expr(in_group_by)
        return expr / expr.sum().over(group_by)


def is_letters(in_expr: IntoExpr, allow_spaces: bool = False) -> pl.Expr:
    """Test whether each string is alphabetic.

    Parameters
    ----------
    in_expr : IntoExpr
        Column name or expression holding the strings to test.
    allow_spaces : bool, default False
        Count a space as a letter, so multi-word strings such as n-grams pass.

    Returns
    -------
    pl.Expr
        Boolean expression, true for a string holding at least one letter and
        nothing beyond letters, apostrophes and hyphens. False for anything
        else, the empty string included. A null stays null.

    Notes
    -----
    "Letter" is the Unicode category, so Greek, Cyrillic and Han count, as do
    accents written as a separate combining mark. Apostrophes (`'` or `’`) and
    hyphens (as in *don't* or *co-op*) count as long as there is at least one
    letter in the string.
    """
    expr = as_expr(in_expr)
    others = "[-'’ ]" if allow_spaces else "[-'’]"
    return expr.str.contains(
        rf"\A{others}*(?:\p{{L}}\p{{M}}*)(?:{others}|\p{{L}}\p{{M}}*)*\z"
    )
