from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from typing import Optional

import polars as pl

from ._internal import Match, Opcode, OpcodeMatcher, Span
from .cqp_parser import cqp
from .search import LazySearchResults, SearchResults
from .simple_parser import simple_to_cqp
from .utils import as_eager, check_columns

__all__ = ["search", "search_cqp", "Match", "Span"]

DEFAULT_CHUNK_TOKENS = 10_000_000
DEFAULT_FILE_ID = "file_id"


def _col_name(i: int) -> str:
    return f"_{i}"


def _compute_masks(
    lf: pl.LazyFrame, opcodes: list[Opcode], keep: Sequence[str] = ()
) -> pl.LazyFrame:
    """Find token positions where a (sub-)query might potentially match."""
    for pc in range(len(opcodes)):
        lf = _propagate_masks(lf, opcodes, pc)
    return lf.select([_col_name(i) for i in range(len(opcodes))] + list(keep))


def _propagate_masks(lf: pl.LazyFrame, opcodes: list[Opcode], pc: int) -> pl.LazyFrame:
    """Propagate token masks backwards through the NFA."""
    if _col_name(pc) not in lf.collect_schema().names():
        match opcodes[pc]:
            case Opcode.Token(expr):
                expr = pl.Expr.deserialize(expr)
                lf = lf.with_columns(expr.fill_null(False).alias(_col_name(pc)))
            case Opcode.Match() | Opcode.Skip() | Opcode.Fail():
                lf = lf.with_columns(pl.lit(True).alias(_col_name(pc)))
            case (
                Opcode.PushVar()
                | Opcode.PopVar()
                | Opcode.BindVar(_)
                | Opcode.UnBindVar()
            ):
                lf = _propagate_masks(lf, opcodes, pc + 1)
                lf = lf.with_columns(pl.col(_col_name(pc + 1)).alias(_col_name(pc)))
            case Opcode.Jump(offset):
                lf = _propagate_masks(lf, opcodes, pc + offset)
                lf = lf.with_columns(
                    pl.col(_col_name(pc + offset)).alias(_col_name(pc))
                )
            case Opcode.Split(offset1, offset2):
                lf = _propagate_masks(lf, opcodes, pc + offset1)
                lf = _propagate_masks(lf, opcodes, pc + offset2)
                lf = lf.with_columns(
                    (
                        pl.col(_col_name(pc + offset1))
                        | pl.col(_col_name(pc + offset2))
                    ).alias(_col_name(pc))
                )
            case _:
                raise RuntimeError(f"Unknown opcode {opcodes[pc]}")
    return lf


def _check_bindings(opcodes: list[Opcode]) -> list[str]:
    """Collect the variables the program binds, in the order it binds them."""
    seen: list[str] = []
    for op in opcodes:
        if isinstance(op, Opcode.BindVar):
            if op._0 in seen:
                raise ValueError(f"Duplicate variable binding: ${op._0}")
            seen.append(op._0)
    return seen


def _compile(cqp_query: str) -> tuple[list[Opcode], list[str]]:
    """The compiled program for `cqp_query` and the variables it binds."""
    opcodes = cqp(cqp_query)
    opcodes.append(Opcode.Match())
    return opcodes, _check_bindings(opcodes)


def _collect_masks(
    lf: pl.LazyFrame, opcodes: list[Opcode], file_id_column: Optional[str]
) -> tuple[list[pl.Series], Optional[pl.Series]]:
    """Run `_compute_masks`, giving one mask per opcode plus the file ids to respect."""
    keep = [] if file_id_column is None else [file_id_column]
    # Rechunked so the matcher's positional lookups stay O(1).
    mask_df = _compute_masks(lf, opcodes, keep).collect(engine="streaming").rechunk()
    masks = [mask_df.get_column(_col_name(pc)) for pc in range(len(opcodes))]
    file_ids = None if file_id_column is None else mask_df.get_column(file_id_column)
    return masks, file_ids


def _partition_files(
    lf: pl.LazyFrame, file_id_column: str, chunk_tokens: int
) -> pl.LazyFrame:
    """Group corpus files into chunks."""
    return (
        lf.group_by(file_id_column, maintain_order=True)
        .agg(pl.len().alias("_len"))
        .with_row_index("_file")
        .with_columns(_offset=pl.col("_len").cum_sum() - pl.col("_len"))
        .with_columns(_chunk=pl.col("_offset") // chunk_tokens)
    )


def _check_contiguous(
    file_ids: pl.Series, chunk_files: pl.DataFrame, file_id_column: str
) -> None:
    """Verify that a materialized chunk holds exactly the file runs the plan gave it."""
    runs = file_ids.rle().struct.unnest()
    if (
        runs["value"].to_list() != chunk_files[file_id_column].to_list()
        or runs["len"].to_list() != chunk_files["_len"].to_list()
    ):
        raise ValueError(
            f"searching a LazyFrame requires all tokens with the same "
            f"{file_id_column!r} to sit together, but the corpus interleaves "
            f"its files; sort it by {file_id_column!r} first"
        )


def _relative_matches(
    matches: list[Match],
    chunk_files: pl.DataFrame,
    file_id_column: str,
    variables: list[str],
) -> pl.DataFrame:
    """One row per match, with spans rebased to offsets within their file."""
    ends = chunk_files["_len"].cum_sum()
    file_starts = (ends - chunk_files["_len"]).to_list()
    ends = ends.to_list()
    file_index = [bisect_right(ends, m.span.start) for m in matches]
    bases = [file_starts[i] for i in file_index]

    data = {
        file_id_column: chunk_files[file_id_column].gather(file_index),
        "_file": chunk_files["_file"].gather(file_index),
        "start": pl.Series(
            [m.span.start - base for m, base in zip(matches, bases)], dtype=pl.UInt32
        ),
        "end": pl.Series(
            [m.span.end - base for m, base in zip(matches, bases)], dtype=pl.UInt32
        ),
    }
    if variables:
        span_type = pl.Struct({"start": pl.UInt32, "end": pl.UInt32})
        data["bindings"] = pl.Series(
            [
                {
                    name: (
                        None
                        if name not in m.bindings
                        else {
                            "start": m.bindings[name].start - base,
                            "end": m.bindings[name].end - base,
                        }
                    )
                    for name in variables
                }
                for m, base in zip(matches, bases)
            ],
            dtype=pl.Struct({name: span_type for name in variables}),
        )
    return pl.DataFrame(data)


def _search_eager(
    df: pl.DataFrame, cqp_query: str, query: str, file_id_column: Optional[str]
) -> Optional[SearchResults]:
    """Search an eager corpus, which fits in memory whole."""
    df = as_eager(df)
    if file_id_column == DEFAULT_FILE_ID and file_id_column not in df.columns:
        file_id_column = None
    if file_id_column is not None:
        check_columns(df, [file_id_column], param="file_id_column")
    if df.is_empty():
        return None

    opcodes, variables = _compile(cqp_query)
    masks, file_ids = _collect_masks(df.lazy(), opcodes, file_id_column)
    matches = OpcodeMatcher(opcodes, masks, file_ids).matchall()
    if not matches:
        return None
    return SearchResults(df, query, matches, variables, file_id_column)


def _search_lazy(
    lf: pl.LazyFrame,
    cqp_query: str,
    query: str,
    file_id_column: Optional[str],
    chunk_tokens: int,
) -> Optional[LazySearchResults]:
    """Search a lazy corpus one chunk of whole files at a time."""
    if file_id_column is None:
        raise ValueError(
            "searching a LazyFrame processes the corpus in chunks of whole "
            "files, so file_id_column must name the column grouping tokens "
            "by file, e.g. file_id_column='file_id'"
        )
    check_columns(lf, [file_id_column], param="file_id_column")
    if (
        not isinstance(chunk_tokens, int)
        or isinstance(chunk_tokens, bool)
        or chunk_tokens < 1
    ):
        raise ValueError(
            f"chunk_tokens must be a positive integer, got {chunk_tokens!r}"
        )

    opcodes, variables = _compile(cqp_query)

    files = _partition_files(lf, file_id_column, chunk_tokens).collect(
        engine="streaming"
    )
    parts = []
    for _, chunk_files in files.group_by("_chunk", maintain_order=True):
        offset = int(chunk_files["_offset"][0])
        length = int(chunk_files["_len"].sum())
        masks, file_ids = _collect_masks(
            lf.slice(offset, length), opcodes, file_id_column
        )
        # to keep pyrefly happy
        assert file_ids is not None
        _check_contiguous(file_ids, chunk_files, file_id_column)
        matches = OpcodeMatcher(opcodes, masks, file_ids).matchall()
        if matches:
            parts.append(
                _relative_matches(matches, chunk_files, file_id_column, variables)
            )
    if not parts:
        return None
    return LazySearchResults(
        lf, query, pl.concat(parts), variables, file_id_column, files
    )


def _search(
    df: pl.DataFrame | pl.LazyFrame,
    cqp_query: str,
    query: str,
    file_id_column: Optional[str],
    chunk_tokens: int,
) -> Optional[SearchResults | LazySearchResults]:
    """Search `df` for the compiled `cqp_query`, reporting it as `query`."""
    if isinstance(df, pl.LazyFrame):
        return _search_lazy(df, cqp_query, query, file_id_column, chunk_tokens)
    return _search_eager(df, cqp_query, query, file_id_column)


def search_cqp(
    df: pl.DataFrame | pl.LazyFrame,
    query: str,
    file_id_column: Optional[str] = DEFAULT_FILE_ID,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Optional[SearchResults | LazySearchResults]:
    """Search corpus using a CQP-style query.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Corpus to be searched.
    query: str
        CQP query string
    file_id_column : str, optional
        Column name holding file ids; matches won't span a change in its
        value. The default "file_id" binds only when the corpus has that
        column; None searches the whole corpus regardless (eager only -- a
        LazyFrame always needs the column to chunk by).
    chunk_tokens : int, default 10_000_000
        Number of tokens to aim for per chunk when searching a LazyFrame.
        Ignored for an eager corpus.

    Returns
    -------
    SearchResults or LazySearchResults or None
        Result of the search  (lazy in, lazy out) or None with no matches.

    Raises
    ------
    ValueError
        If `df` is not a polars frame, does not have `file_id_column`, or is
        a LazyFrame given without one
    ParseException
        If there's an error in the query
    RuntimeError
        If there's an internal error in the search procedure

    Notes
    -----
    For CQP query syntax documentation, see the CQP documentation.

    Examples
    --------
    >>> search_cqp(corpus, '[word="fox"]')
    >>> search_cqp(corpus, '[pos="NN.*"]+ [pos="VB.*"]')
    >>> search_cqp(pl.scan_parquet("bnc.parquet"), '[word="fox"]',
    ...            file_id_column="file_id")

    """
    return _search(df, query, query, file_id_column, chunk_tokens)


def search(
    df: pl.DataFrame | pl.LazyFrame,
    query: str,
    token_column: str = "token",
    pos_column: str = "pos",
    lemma_column: str = "lemma",
    file_id_column: Optional[str] = DEFAULT_FILE_ID,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> Optional[SearchResults | LazySearchResults]:
    """Search corpus using simple query language (BNCweb-style).

    This function uses a simple query syntax similar to BNCweb's. Queries
    are case-insensitive throughout and support wildcards, alternatives,
    word sequences, POS tags, and lemma searches.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Corpus to be searched.
    query : str
        Simple query string (BNCweb-style syntax)
    token_column : str, optional
        Column name for token searches (default: "token")
    pos_column : str, optional
        Column name for POS tag searches (default: "pos")
    lemma_column : str, optional
        Column name for lemma searches (default: "lemma")
    file_id_column : str, optional
        Column name holding file ids; matches won't span a change in its
        value. The default "file_id" binds only when the corpus has that
        column; None searches the whole corpus regardless (eager only -- a
        LazyFrame always needs the column to chunk by).
    chunk_tokens : int, default 10_000_000
        Number of tokens to aim for per chunk when searching a LazyFrame.
        Ignored for an eager corpus.

    Returns
    -------
    SearchResults or LazySearchResults or None
        Result of the search (lazy in, lazy out) or None if no matches
        found

    Raises
    ------
    ValueError
        If `df` is not a polars frame, does not have `file_id_column`, or is
        a LazyFrame given without one
    ParseException
        If there's an error in the query syntax
    RuntimeError
        If there's an internal error in the search procedure

    Notes
    -----
    Simple query syntax supports:

    - **Basic words**: `fox` matches "fox", "Fox", "FOX" (case-insensitive)
    - **Wildcards**:
      - `?` for single character: `s?ng` → sing, sang, song
      - `*` for zero or more: `*able` → able, table, capable
      - `+` for one or more: `+able` → table, capable (not "able")
    - **Alternatives**: `[car,truck]`, `neighbo[u,]r`
    - **Sequences**: `quick brown fox`
    - **Gaps**:
      - `*` for optional token: `eat * up` → "eat up", "eat it up"
      - `+` for required token: `eat + up` → "eat it up" (not "eat up")
    - **POS tags**: `word_TAG` for word+POS, `_TAG` for POS only
      - `lights_NN2` → "lights" tagged as NN2
      - `*ly_AJ0` → adjectives ending in "-ly"
      - `_PNX` → any reflexive pronoun
    - **Lemmas**: `{lemma}`, `{lemma/POS}`, or `{lemma}_TAG` for lemma searches
      - `{light}` → all forms of "light"
      - `{light/V}` → verbal forms using simplified POS (V, N, A, etc.)
      - `{walk}_VBD` → lemma "walk" with exact POS tag VBD
      - `{be}_V*` → lemma "be" with any verb POS tag
      - `{eat} * up` → lemma "eat" followed by "up"
    - **Variable bindings**: `$varname: pattern` to capture subpatterns
      - `$target: fox` → capture "fox" position
      - `$phrase: (quick brown)` → capture multi-token span
      - Access via `match.bindings[varname]`
    - **Escaping**: `\\?` for literal question mark

    For CQP queries with advanced features, use `search_cqp()` instead.

    Examples
    --------
    >>> search(corpus, "fox")  # Find "fox" (case-insensitive)
    >>> search(corpus, "s?ng")  # Find sing, sang, song
    >>> search(corpus, "*able")  # Find words ending in "able"
    >>> search(corpus, "[car,truck]")  # Find either "car" or "truck"
    >>> search(corpus, "quick brown fox")  # Find exact sequence
    >>> search(corpus, "fox + over")  # "fox" followed by any word, then "over"

    >>> # POS tag searches
    >>> search(corpus, "lights_NN2")  # "lights" as plural noun
    >>> search(corpus, "*ly_AJ0")  # Adjectives ending in "-ly"
    >>> search(corpus, "_PNX")  # Any reflexive pronoun

    >>> # Lemma searches
    >>> search(corpus, "{light}")  # All forms of lemma "light"
    >>> search(corpus, "{light/V}")  # Verbal forms (simplified POS)
    >>> search(corpus, "{walk}_VBD")  # Lemma "walk" with exact POS tag
    >>> search(corpus, "{eat} * up")  # Lemma "eat" followed by "up"

    >>> # Variable bindings
    >>> results = search(corpus, "$verb: {walk}")
    >>> match = results.matches[0]
    >>> verb_span = match.bindings["verb"]
    >>> verb_text = corpus["token"][verb_span.start:verb_span.end]

    >>> # Search in a different column
    >>> search(corpus, "NN*", token_column="pos")  # Find noun POS tags

    >>> # Keep matches from spanning file boundaries
    >>> search(corpus, "quick brown", file_id_column="file_id")

    """
    cqp_query = simple_to_cqp(query, token_column, pos_column, lemma_column)
    return _search(df, cqp_query, query, file_id_column, chunk_tokens)
