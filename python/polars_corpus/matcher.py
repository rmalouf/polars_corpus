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
    """
    Find every place in a corpus where a CQP query matches.

    A CQP query is a sequence of token patterns. Each one is written in
    brackets as a test against the corpus's columns, so `[pos="NN.*"]` matches
    a single token whose `pos` column matches that regular expression. Within
    a bracket, `&` and `|` combine tests and `!=` negates one. Between
    brackets, the patterns match consecutive tokens, and quantifiers such as
    `+` and `{2,3}` apply to them as they do in a simple query.

    Because the tests name columns directly, a CQP query can use any
    annotation the corpus carries, not just the token, tag and lemma that
    `search` has shorthands for. Anything a simple query can express, a CQP
    query can express too: `search` compiles to CQP and runs the result.

    Unlike simple queries, these tests are case-sensitive by default. Add `%c`
    to one to ignore case, as in `[token="fox"%c]`.

    Parameters
    ----------
    df : DataFrame | LazyFrame
        Corpus to search. A LazyFrame is searched a chunk of whole files at a
        time, so the corpus is never held in memory at once.
    query : str
        CQP query, e.g. `[pos="NN.*"] [lemma="be"]`. See QUERY_LANGUAGE.md for
        the syntax.
    file_id_column : str, optional
        Column holding file ids, which mark where one text ends and the next
        begins. No match crosses a change in its value. The default of
        "file_id" applies only if the corpus has that column, so a corpus
        without one still searches. Pass None to search the corpus as one
        continuous run of tokens; this works for eager corpora only, since a
        LazyFrame needs the column to chunk by.
    chunk_tokens : int, default 10_000_000
        Tokens to aim for per chunk when searching a LazyFrame. Lower it to
        use less memory, raise it for fewer passes over the corpus. Ignored
        for an eager corpus.

    Returns
    -------
    SearchResults or LazySearchResults or None
        The matches: `SearchResults` for a DataFrame, `LazySearchResults` for
        a LazyFrame. None if the query matched nothing, so check the result
        before using it.

    Raises
    ------
    ValueError
        If `df` is not a Polars DataFrame or LazyFrame, is missing
        `file_id_column`, or is a LazyFrame given `file_id_column=None`; if
        `chunk_tokens` is not a positive integer; or if `query` binds the same
        variable twice.
    lark.exceptions.LarkError
        If `query` is not a well-formed CQP query.

    See Also
    --------
    search : The same search, written in the simple query language.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.search_cqp(corpus, '[token="fox"%c]')  # %c to ignore case
    >>> plc.search_cqp(corpus, '[pos="NN.*"]+ [pos="VB.*"]')  # Nouns, then a verb
    >>> plc.search_cqp(corpus, '[pos!="NN.*"]')  # Anything but a noun
    >>> # A column that the simple query shorthands cannot reach:
    >>> plc.search_cqp(corpus, '[speaker="A" & lemma="think"]')
    >>> # Out of core, over a corpus too large to hold in memory:
    >>> plc.search_cqp(pl.scan_parquet("bnc.parquet"), '[token="fox"%c]')
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
    """
    Find every place in a corpus where a simple (BNCweb-style) query matches.

    A simple query is written the way the words appear. `fox` finds the word;
    `quick brown fox` finds the three of them in sequence.

    Inside a word, `?` stands for one character, and `*` and `+` stand for any
    number of them. So `s?ng` finds *sing*, *sang* and *song*, and `*able`
    finds words ending in *able*. Between words, `*` and `+` stand for whole
    tokens instead: `eat * up` finds both *eat up* and *eat it up*, while
    `eat + up` finds only *eat it up*.

    `_TAG` constrains a token's tag, `{lemma}` constrains its lemma, and
    `$name:` captures part of the match so that `concordance` can give it a
    column. Every test ignores case.

    The query is compiled to CQP and run by `search_cqp`. Anything this syntax
    cannot express can be written as a CQP query instead.

    Parameters
    ----------
    df : DataFrame | LazyFrame
        Corpus to search. A LazyFrame is searched a chunk of whole files at a
        time, so the corpus is never held in memory at once.
    query : str
        Simple query, e.g. `quick brown fox` or `{light}_V*`. See
        [Simple query language](simple_query.md) for the full syntax.
    token_column : str, default "token"
        Column a bare word in the query is matched against.
    pos_column : str, default "pos"
        Column the `_TAG` part of a query is matched against.
    lemma_column : str, default "lemma"
        Column a `{lemma}` query is matched against. Only read by a query that
        asks for a lemma, so an unlemmatized corpus still searches.
    file_id_column : str, optional
        Column holding file ids, which mark where one text ends and the next
        begins. No match crosses a change in its value. The default of
        "file_id" applies only if the corpus has that column, so a corpus
        without one still searches. Pass None to search the corpus as one
        continuous run of tokens; this works for eager corpora only, since a
        LazyFrame needs the column to chunk by.
    chunk_tokens : int, default 10_000_000
        Tokens to aim for per chunk when searching a LazyFrame. Lower it to
        use less memory, raise it for fewer passes over the corpus. Ignored
        for an eager corpus.

    Returns
    -------
    SearchResults or LazySearchResults or None
        The matches: `SearchResults` for a DataFrame, `LazySearchResults` for
        a LazyFrame. None if the query matched nothing, so check the result
        before using it.

    Raises
    ------
    ValueError
        If `df` is not a Polars DataFrame or LazyFrame, is missing
        `file_id_column`, or is a LazyFrame given `file_id_column=None`; if
        `chunk_tokens` is not a positive integer; or if `query` binds the same
        variable twice.
    lark.exceptions.LarkError
        If `query` is not a well-formed simple query.
    polars.exceptions.ColumnNotFoundError
        If the corpus has no column the query asks for -- a `_TAG` query
        against a corpus with no `pos_column`, say. Point the `*_column`
        arguments at the columns the corpus does have.

    See Also
    --------
    search_cqp : The same search, written as a CQP query.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> plc.search(corpus, "fox")
    >>> plc.search(corpus, "quick brown fox")  # A sequence of words
    >>> plc.search(corpus, "*ly_AJ0")  # Adjectives ending in "-ly"
    >>> plc.search(corpus, "{eat} * up")  # Forms of "eat", then "up"
    >>> # A query written over a differently annotated corpus:
    >>> plc.search(corpus, "{light}_V*", lemma_column="headword", pos_column="c5")
    >>> # $name: captures part of the match, for concordance() to column off:
    >>> plc.search(corpus, "$adj: _AJ0 $noun: _NN1")
    """
    cqp_query = simple_to_cqp(query, token_column, pos_column, lemma_column)
    return _search(df, cqp_query, query, file_id_column, chunk_tokens)
