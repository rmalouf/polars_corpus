from __future__ import annotations

import random
from collections.abc import Sequence
from difflib import get_close_matches
from typing import TYPE_CHECKING, Optional, Self

import polars as pl
from polars._typing import IntoExprColumn

from ._internal import Match, Span, py_concordance, py_kwic, spans_to_chunks
from .utils import as_eager, check_columns, output_name

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["SearchResults", "LazySearchResults", "concordance", "collocates"]


def _select(
    df: pl.DataFrame,
    expr: IntoExprColumn | list[IntoExprColumn],
    param: str = "expr",
) -> pl.DataFrame:
    """The columns `expr` names, with a bad name reported against the corpus.

    `expr` may name several columns -- a list, a regex, a selector -- so it is
    resolved rather than checked ahead of time, and the error a missing column
    raises is caught here, where it can say which corpus lacks it.
    """
    try:
        return df.select(expr)
    except pl.exceptions.ColumnNotFoundError as err:
        items = expr if isinstance(expr, list) else [expr]
        roots: list[str] = []
        for item in items:
            if isinstance(item, str):
                roots.append(item)
            elif isinstance(item, pl.Expr):
                roots.extend(item.meta.root_names())
        check_columns(df, roots, param=param)
        raise ValueError(f"the corpus cannot evaluate {param}: {err}") from err


def _check_variables(value: object, available: list[str]) -> list[str]:
    """Check that `value` names variables the query bound, keeping their order."""
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, (list, tuple)):
        raise ValueError(
            f"bindings must be True, False, a variable name, or a list of them; "
            f"got {type(value).__name__}"
        )
    for name in names:
        if name not in available:
            if not available:
                raise ValueError(
                    f"the query bound no variables, so there is no {name!r} to "
                    f"show; write ${name}: in the query to bind it"
                )
            close = get_close_matches(str(name), available, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(
                f"the query bound no variable {name!r}. It bound: "
                f"{', '.join(available)}.{hint}"
            )
    return list(dict.fromkeys(names))


def _check_count(value: object, param: str, hint: str = "") -> int:
    """Check that `value` is a count: an integer, zero or more."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(
            f"{param} must be a non-negative integer, got {value!r}.{hint}"
        )
    return value


def _build_conc(
    df: pl.DataFrame,
    matches: list[Match],
    expr: IntoExprColumn | list[IntoExprColumn],
    window: int,
    chunk_column: Optional[str],
    metadata: Optional[list[str]],
    names: list[str],
    file_ids: Optional[pl.Series],
) -> pl.DataFrame:
    """One concordance frame from an eager frame and matches indexing into it."""
    metadata_df = None if metadata is None else df.select(metadata)
    if chunk_column is not None:
        chunk_tags = df.get_column(chunk_column)
        # The tags are read as strings; a dictionary-encoded column holds the
        # same ones and is worth the cast to get at them.
        if isinstance(chunk_tags.dtype, (pl.Categorical, pl.Enum)):
            chunk_tags = chunk_tags.cast(pl.String)
        return py_concordance(
            _select(df, expr), matches, chunk_tags, metadata_df, names, file_ids
        )
    return py_kwic(
        _select(df, expr), matches, window, window, metadata_df, names, file_ids
    )


def _needed_columns(
    expr: IntoExprColumn | list[IntoExprColumn],
    metadata: Optional[list[str]],
    chunk_column: Optional[str],
    file_id_column: str,
) -> Optional[list[str]]:
    """Columns a chunk must materialize, or None when `expr` defeats name analysis."""
    items = expr if isinstance(expr, list) else [expr]
    names: list[str] = []
    for item in items:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, pl.Expr):
            roots = item.meta.root_names()
            # A regex reference or selector names no root, resolving against
            # the schema instead; materialize everything rather than guess.
            if not roots:
                return None
            names.extend(roots)
        else:
            return None
    names.extend(metadata or [])
    if chunk_column is not None:
        names.append(chunk_column)
    names.append(file_id_column)
    return list(dict.fromkeys(names))


class _SearchResultsBase:
    """Behavior shared by SearchResults and LazySearchResults.

    A subclass stores the corpus and the matches its own way and provides the
    hooks; everything else here reads them.
    """

    _query: str
    _variables: list[str]

    def _frame(self) -> pl.DataFrame | pl.LazyFrame:
        """The corpus, in whatever form it is held."""
        raise NotImplementedError

    def _corpus_size(self) -> int:
        """Number of tokens in the corpus."""
        raise NotImplementedError

    def _concordance(
        self,
        expr: IntoExprColumn | list[IntoExprColumn],
        window: int,
        chunk_column: Optional[str],
        metadata: Optional[list[str]],
        names: list[str],
    ) -> pl.DataFrame:
        """Build the concordance frame; arguments arrive validated."""
        raise NotImplementedError

    def _take(self, indices: Sequence[int]) -> Self:
        """These results cut down to the matches at `indices`, in that order."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of matches found."""
        raise NotImplementedError

    @property
    def variables(self) -> list[str]:
        """Names the query's `$name:` bindings capture, in the order it binds them.

        A binding the query can skip -- one under `?`, or in an alternative not
        taken -- is named here all the same, and is null in the concordance for
        the matches that did not capture it.

        Examples
        --------
        >>> results = plc.search(corpus, "$det: the $noun: _NN1")
        >>> results.variables
        ['det', 'noun']
        """
        return self._variables

    def __repr__(self) -> str:
        es = "" if len(self) == 1 else "es"
        return f"{type(self).__name__}<'{self._query}'; {len(self):,} match{es}>"

    def concordance(
        self,
        expr: IntoExprColumn | list[IntoExprColumn] = "token",
        window: int = 0,
        chunk_column: Optional[str] = None,
        metadata: Optional[str | list[str]] = None,
        bindings: bool | str | list[str] = True,
    ) -> pl.DataFrame:
        """
        Lay the matches out as a KWIC (keyword in context) concordance.

        Returns one row per match. The matched words go in a column named
        after `expr`, the words before them in `{expr}_left_context`, and the
        words after them in `{expr}_right_context`. All three hold a list of
        tokens rather than a joined string, so they can be filtered and
        aggregated like any other list column.

        Parameters
        ----------
        expr : IntoExprColumn or list of IntoExprColumn, default "token"
            Column name or expression to read the concordance text from
            (e.g. token or lemma). Pass a list to get one set of columns per
            name, so that words and tags can be shown side by side.
        window : int, default 0
            Words of context to take on each side of a match. The default of 0
            takes none, and the context columns are then left out. Ignored
            when `chunk_column` is given.
        chunk_column : str, optional
            Column of BIO tags, marking each chunk with "B" on its first token
            and "I" on the rest. Context then runs to the edges of the chunk
            holding the match instead of counting a fixed number of words:
            back to the "B" that opened the chunk, and forward to the last "I"
            before the next one. Pass a sentence tag column to get the whole
            sentence a match sits in. `window` is ignored.
        metadata : str or list of str, optional
            Column(s) of the corpus to carry through to each row as ordinary
            values rather than lists, e.g. `file_id` or `category`. The value
            is read off the match's first token.
        bindings : bool, str or list of str, default True
            Which of the query's `$name:` captures to give a column of their
            own. True takes all of them, which does nothing if the query
            captured none. False takes none. A name, or a list of names, takes
            just those. See `variables` for what a query captured.

        Returns
        -------
        pl.DataFrame
            One row per match, in the order the results hold them. That is
            corpus order, unless `sample` or `shuffle` has reordered them. For
            each column `expr` names:

            - `{column}` : the matched words, as a list
            - `{column}_left_context` : the words before the match, left out
              when `window` is 0 and no `chunk_column` is given
            - `{column}_right_context` : the words after the match, left out
              under the same condition
            - `{column}_{name}` : the words captured by each `$name:`

            Each name in `metadata` adds one more column, holding a single
            value rather than a list.

        Raises
        ------
        ValueError
            If `window` is negative; if the corpus is missing a column named
            in `metadata` or `chunk_column`, or `chunk_column` does not hold
            strings; or if `bindings` names something the query did not
            capture.

        Notes
        -----
        Context stops at a file boundary when the results know their
        `file_id_column`, as results of a search that named one do. Without it
        context is taken from the corpus in the order it is held, so it will
        run from the end of one file into the start of the next.

        A binding column holds an empty list where the variable matched no
        token, as an optional one can, and a null where the match went down a
        branch of the query that never bound it at all.

        Examples
        --------
        >>> results.concordance("token", window=3)
        >>> results.concordance("token")  # The matches on their own
        >>> results.concordance(["token", "pos"], window=5)  # Words and tags
        >>> # Whole sentences of context, labelled with where they came from:
        >>> results.concordance(
        ...     "token", chunk_column="sentence_tag", metadata=["file_id", "category"]
        ... )

        >>> # A column per capture: token_adj and token_noun
        >>> results = plc.search(corpus, "$adj: _AJ0 $noun: _NN1")
        >>> results.concordance("token", window=5)
        >>> results.concordance("token", bindings="adj")  # Just the one
        """
        if bindings is True:
            names = list(self.variables)
        elif bindings is False:
            names = []
        else:
            names = _check_variables(bindings, self.variables)

        if isinstance(metadata, str):
            metadata = [metadata]
        if metadata is not None:
            check_columns(self._frame(), metadata, param="metadata")

        if chunk_column is not None:
            check_columns(self._frame(), [chunk_column], param="chunk_column")
            dtype = self._frame().lazy().collect_schema()[chunk_column]
            if dtype != pl.String and not isinstance(dtype, (pl.Categorical, pl.Enum)):
                raise ValueError(
                    f"chunk_column must hold the chunk tags 'B', 'I' and 'O' as "
                    f"strings, but {chunk_column!r} holds {dtype}"
                )
        else:
            window = _check_count(window, "window", " Use window=0 for no context.")

        return self._concordance(expr, window, chunk_column, metadata, names)

    def collocates(
        self,
        expr: IntoExprColumn = "token",
        window: int = 5,
        min_freq: int = 5,
        freqs_name: str = "freqs",
    ) -> pl.DataFrame:
        """
        Count the words that occur near the matches.

        Takes the `window` words on each side of every match and counts how
        often each word occurs across all of them. This is the same context
        `concordance` shows; the matched words themselves are not counted.

        Each count comes with the three other frequencies an association
        measure needs, so the result can be passed straight to one to find the
        words that occur near the matches more often than their own corpus
        frequency would predict. See [Association metrics](assoc.md).

        Parameters
        ----------
        expr : IntoExprColumn, default "token"
            Column name or expression to read the collocates from (e.g. token
            or lemma). It must name a single column.
        window : int, default 5
            Words to take on each side of a match. Must be at least 1.
        min_freq : int, default 5
            Number of times a word must occur in the windows to be reported.
            Association measures are unstable for rare pairs, so the default
            drops them. Pass 0 to keep them all.
        freqs_name : str, default "freqs"
            Name for the struct column the counts come back in.

        Returns
        -------
        pl.DataFrame
            One row per distinct word found in the windows, in no particular
            order:

            - `collocate` : the word
            - `freqs` : a struct of four counts, named as the association
              measures expect:

                - `f12` : times the word fell in a window
                - `f1` : number of window positions available
                - `f2` : the word's frequency in the whole corpus
                - `n` : number of tokens in the corpus

        Raises
        ------
        ValueError
            If `window` is less than 1, `min_freq` is negative, or `expr`
            names more than one column.

        Notes
        -----
        The struct has the same layout `crosstab` produces, so the association
        measures take it as it comes.

        `f1` is `len(results) * window * 2`, the number of positions the
        windows could hold. A window that runs off the end of a file holds
        fewer, so `f1` is a slight overcount.

        A word in the windows of two nearby matches is counted once for each,
        so overlapping windows count their shared words twice. A word's `f12`
        can therefore come out higher than its corpus frequency `f2`.

        Examples
        --------
        >>> results.collocates("token")
        >>> results.collocates("token", window=3, min_freq=10)
        >>> # Rank the collocates by how strongly they attract the search term:
        >>> collocs = results.collocates("token")
        >>> collocs.with_columns(ll=pl.col("freqs").corpus.loglik()).sort(
        ...     "ll", descending=True
        ... )
        """
        if _check_count(window, "window") == 0:
            raise ValueError("window must be at least 1 to have any collocates in it")
        _check_count(min_freq, "min_freq", " Use min_freq=0 to keep them all.")
        if isinstance(expr, (list, tuple)):
            raise ValueError(
                "expr must name a single column to collocate with, not a list of "
                "them; call collocates() once per column instead"
            )
        name = output_name(expr)
        conc = self.concordance(expr, window=window, bindings=False)
        return (
            conc.lazy()
            .select(
                collocate=pl.col(f"{name}_left_context")
                .list.concat(f"{name}_right_context")
                .explode(empty_as_null=False)
            )
            # A null token is not a collocate either.
            .drop_nulls()
            .group_by("collocate")
            .len(name="f12")
            .filter(pl.col("f12") >= min_freq)
            .join(
                self._frame().lazy().group_by(expr).len(name="f2"),
                left_on="collocate",
                right_on=name,
                how="left",
            )
            .with_columns(n=self._corpus_size(), f1=len(self) * window * 2)
            .select("collocate", pl.struct("f12", "f1", "f2", "n").alias(freqs_name))
            .collect(engine="streaming")
        )

    def view(
        self,
        expr: IntoExprColumn | list[IntoExprColumn] = "token",
        window: int = 5,
        chunk_column: Optional[str] = None,
        metadata: Optional[str | list[str]] = None,
        page_size: int = 25,
    ) -> None:
        """
        Browse the concordance in a Jupyter notebook, a page at a time.

        Builds the same lines `concordance` builds and displays them in a
        widget that pages, sorts and filters. Returns nothing; it draws in the
        notebook.

        Parameters
        ----------
        expr : IntoExprColumn or list of IntoExprColumn, default "token"
            Column name or expression to read the concordance text from
            (e.g. token or lemma). Where it names several, the first is the
            one shown.
        window : int, default 5
            Words of context to show on each side of a match. Pass 0 for none.
            Ignored when `chunk_column` is given.
        chunk_column : str, optional
            Column of BIO tags, to take context out to the edges of the chunk
            holding each match rather than by a fixed count, as in
            `concordance`.
        metadata : str or list of str, optional
            Column(s) of the corpus to carry into the underlying concordance,
            e.g. `file_id` or `category`. The viewer draws only the match and
            its context, so these are not yet shown.
        page_size : int, default 25
            Concordance lines to draw per page.

        See Also
        --------
        concordance : The same lines as a DataFrame, to compute over.

        Examples
        --------
        >>> results.view()
        >>> results.view("token", window=10, page_size=50)
        >>> # Whole sentences of context rather than a fixed window:
        >>> results.view("token", chunk_column="sentence_tag")
        """
        from .view import ConcordanceWidget

        conc = self.concordance(
            expr,
            window=window,
            chunk_column=chunk_column,
            metadata=metadata,
            bindings=False,
        )
        ConcordanceWidget(conc, page_size=page_size).show()

    def head(self, n: int) -> Self:
        """
        Keep the first `n` matches, dropping the rest.

        "First" in the order the results are currently in: corpus order as a
        search returns them, or the order `sample` or `shuffle` left them in.

        Parameters
        ----------
        n : int
            Matches to keep. Asking for more than there are keeps them all.

        Returns
        -------
        Self
            The same kind of results, cut down to those matches.

        Raises
        ------
        ValueError
            If `n` is negative.

        Examples
        --------
        >>> results.head(10).concordance("token", window=5)
        """
        _check_count(n, "n")
        if n >= len(self):
            return self
        return self._take(range(n))

    def tail(self, n: int) -> Self:
        """
        Keep the last `n` matches, dropping the rest.

        "Last" in the order the results are currently in: corpus order as a
        search returns them, or the order `sample` or `shuffle` left them in.

        Parameters
        ----------
        n : int
            Matches to keep. Asking for more than there are keeps them all.

        Returns
        -------
        Self
            The same kind of results, cut down to those matches.

        Raises
        ------
        ValueError
            If `n` is negative.

        Examples
        --------
        >>> results.tail(10).concordance("token", window=5)
        """
        _check_count(n, "n")
        if n >= len(self):
            return self
        return self._take(range(len(self) - n, len(self)))

    def sample(self, k: int, seed: Optional[int] = None) -> Self:
        """
        Draw `k` of the matches at random, without replacement.

        Useful when a query has more hits than can be read through. Every
        match is equally likely to be drawn, so the sample is not weighted
        toward the start of the corpus. The matches come back in the order
        drawn, not in corpus order.

        Parameters
        ----------
        k : int
            Matches to draw. Cannot be more than there are.
        seed : int, optional
            Seed for the draw. Set it to get the same sample back every time
            the code is rerun, which is what reporting a result needs. Without
            one, the draw differs each run.

        Returns
        -------
        Self
            The same kind of results, holding the `k` matches drawn.

        Raises
        ------
        ValueError
            If `k` is negative or larger than the number of matches.

        Notes
        -----
        Seeding is local: the global random state is put back afterwards, so
        this does not disturb anything else in the notebook drawing at random.

        Examples
        --------
        >>> # 100 hits to read through, the same 100 on every rerun:
        >>> results.sample(100, seed=42).view()
        """
        _check_count(k, "k")
        if k > len(self):
            raise ValueError(
                f"cannot sample {k:,} of {len(self):,} matches; "
                f"k must be no larger than the number of matches"
            )
        state = random.getstate()
        random.seed(seed)
        try:
            return self._take(random.sample(range(len(self)), k))
        finally:
            random.setstate(state)

    def shuffle(self, seed: Optional[int] = None) -> Self:
        """
        Put the matches in random order, keeping all of them.

        Parameters
        ----------
        seed : int, optional
            Seed for the shuffle. Set it to get the same order back every time
            the code is rerun. Without one, the order differs each run.

        Returns
        -------
        Self
            The same kind of results, with the matches reordered.

        Notes
        -----
        Seeding is local: the global random state is put back afterwards, so
        this does not disturb anything else in the notebook drawing at random.

        Examples
        --------
        >>> results.shuffle(seed=123).head(20)
        """
        return self.sample(len(self), seed)

    def encode(
        self,
        model: SentenceTransformer,
        expr: IntoExprColumn = "token",
        window: int = 5,
        chunk_column: Optional[str] = None,
        metadata: Optional[str | list[str]] = None,
    ) -> pl.DataFrame:
        """Embed each match together with its context.

        Builds a concordance, joins each line back into a single string, and
        encodes it with `model`, so that matches used in similar contexts get
        similar vectors.

        Parameters
        ----------
        model : SentenceTransformer
            Model used to encode the concordance lines.
        expr : IntoExprColumn, default "token"
            Column name or Polars expression holding the text to encode. It
            must name a single column.
        window : int, default 5
            Number of tokens of context on both sides of each match. Pass 0
            to encode the matches on their own. Ignored when chunk_column is
            given.
        chunk_column : str, optional
            Column name defining chunk boundaries, as in `concordance`. When
            given, context extends to the chunk holding the match.
        metadata : str or list[str], optional
            Column name(s) from the source corpus to carry through to the
            result, e.g. file_id or category.

        Returns
        -------
        pl.DataFrame
            One row per match, with the concordance line under the output name
            of `expr`, its embedding in a "vector" column of
            `Array(Float32, dim)`, and a column for each name in metadata.

        Examples
        --------
        >>> from sentence_transformers import SentenceTransformer
        >>> model = SentenceTransformer("all-MiniLM-L6-v2")
        >>> results.encode(model, window=10)
        >>> results.encode(model, chunk_column="sentence_tag", metadata="file_id")
        """
        from .embeddings import encode

        if isinstance(expr, (list, tuple)):
            raise ValueError(
                "expr must name a single column to encode, not a list of them; "
                "call encode() once per column instead"
            )
        name = output_name(expr)
        conc = self.concordance(
            expr,
            window=window,
            chunk_column=chunk_column,
            metadata=metadata,
            bindings=False,
        )
        parts = [
            column
            for column in (f"{name}_left_context", name, f"{name}_right_context")
            if column in conc.columns
        ]
        keep = [metadata] if isinstance(metadata, str) else list(metadata or [])
        return (
            conc.lazy()
            .select(pl.concat_list(parts).list.join(" ").alias(name), *keep)
            .with_columns(vector=encode(model, name))
            .collect()
        )


class SearchResults(_SearchResultsBase):
    """
    The matches found by searching an in-memory corpus.

    Returned by `search` and `search_cqp` when given a DataFrame; there is no
    reason to construct one directly. It holds the corpus, and for each match
    the span of tokens the query covered. The words themselves stay in the
    corpus and are read back out when a method needs them.

    Use `len()` to count the matches, and `concordance` or `collocates` to
    turn them into a frame. `head`, `tail` and `sample` cut them down to a
    readable number, and `shuffle` reorders them; all four return results of
    this same kind, so they can be chained.

    Parameters
    ----------
    df : DataFrame
        Corpus the search ran over. It must be eager: the matches index into
        it by position, and every method here reads its rows.
    query : str
        Query these results came from, shown in their `repr`.
    matches : list of Match
        The matched spans, in corpus order, each with what it captured.
    variables : list of str
        Names the query captured, in the order to show them.
    file_id_column : str, optional
        Column holding file ids. When given, concordance context stops at a
        change in its value, as the matches themselves did during the search.

    See Also
    --------
    LazySearchResults : The same interface over a corpus too large to hold in
        memory.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> results = plc.search(corpus, "quick")
    >>> results
    SearchResults<'quick'; 1 match>
    >>> len(results)
    1
    >>> results.concordance("token", window=3)
    """

    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matches: list[Match],
        variables: list[str],
        file_id_column: Optional[str] = None,
    ) -> None:
        self._df = as_eager(df)
        if file_id_column is not None:
            check_columns(self._df, [file_id_column], param="file_id_column")
        self._query = query
        self._matches = matches
        self._variables = variables
        self._file_id_column = file_id_column

    def __len__(self) -> int:
        """Number of matches found."""
        return len(self._matches)

    @property
    def matches(self) -> list[Match]:
        """The matched spans, in corpus order, each with the variables it bound.

        Positions index into the corpus the search ran over, so a match reads
        back as `corpus[m.span.start : m.span.end]`. `LazySearchResults` has
        no counterpart: it holds its matches as a frame of file-relative spans
        rather than materializing one object apiece.

        Examples
        --------
        >>> results = plc.search(corpus, '$noun: [pos="NN"]')
        >>> results.matches[0].span
        Span(3, 4)
        >>> results.matches[0].bindings["noun"]
        Span(3, 4)
        """
        return self._matches

    def _frame(self) -> pl.DataFrame:
        return self._df

    def _corpus_size(self) -> int:
        return self._df.height

    def _concordance(
        self,
        expr: IntoExprColumn | list[IntoExprColumn],
        window: int,
        chunk_column: Optional[str],
        metadata: Optional[list[str]],
        names: list[str],
    ) -> pl.DataFrame:
        file_ids = (
            None
            if self._file_id_column is None
            else self._df.get_column(self._file_id_column)
        )
        return _build_conc(
            self._df,
            self._matches,
            expr,
            window,
            chunk_column,
            metadata,
            names,
            file_ids,
        )

    def _take(self, indices: Sequence[int]) -> SearchResults:
        return SearchResults(
            self._df,
            self._query,
            [self._matches[i] for i in indices],
            self._variables,
            self._file_id_column,
        )

    def with_spans_as_chunks(self, name: str = "spans") -> pl.DataFrame:
        """
        Write the matches back onto the corpus as a column of BIO tags.

        The reverse of pulling the matches out of the corpus: this marks them
        where they sit. The first token of each match is tagged "B", the rest
        of it "I", and every token no match covers "O". The result is an
        ordinary column, so the matches can be counted, grouped or plotted
        alongside the rest of the corpus.

        Parameters
        ----------
        name : str, default "spans"
            Name for the tag column added to the corpus.

        Returns
        -------
        pl.DataFrame
            The corpus, one row per token as before, with the tag column
            added.

        See Also
        --------
        polars_corpus.with_chunk_index : Number the chunks such a column marks.

        Examples
        --------
        >>> tagged = results.with_spans_as_chunks()
        >>> # One "B" per match, so counting them counts matches per file:
        >>> tagged.group_by("file_id").agg((pl.col("spans") == "B").sum())
        """
        # Extract spans from matches for Rust function
        spans = [match.span for match in self._matches]
        return self._df.with_columns(
            spans_to_chunks(spans, self._df.height).alias(name)
        )


class LazySearchResults(_SearchResultsBase):
    """
    The matches found by searching an out-of-core corpus.

    Returned by `search` and `search_cqp` when given a LazyFrame; there is no
    reason to construct one directly. It has the same methods as
    `SearchResults`, but never holds the corpus in memory. It stores each
    match as an offset within the file the match falls in, and re-reads the
    corpus when a method needs the words. Each read covers only the part of a
    chunk that its matches fall in, so a concordance over a 100M-word corpus
    costs a partial scan rather than the memory to hold the corpus.

    `SearchResults.matches` has no counterpart here. With no corpus in memory,
    there is nothing for match positions to index into.

    Parameters
    ----------
    lf : LazyFrame
        Corpus the search ran over. Tokens sharing a file id must be
        contiguous, as the search itself already checked.
    query : str
        Query these results came from, shown in their `repr`.
    matches : DataFrame
        One row per match: the file id column, `start` and `end` (token
        offsets within the file), `_file` (row index into `files`), and --
        when the query captured variables -- a `bindings` struct column of
        per-variable file-relative spans.
    variables : list of str
        Names the query captured, in the order it captures them.
    file_id_column : str
        Column holding file ids.
    files : DataFrame
        One row per file in corpus order: the file id column, `_file`,
        `_len`, `_offset` (global token offset) and `_chunk` (which chunk of
        the search held it).

    See Also
    --------
    SearchResults : The same interface over a corpus held in memory.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> results = plc.search(pl.scan_parquet("bnc.parquet"), "{eat} * up")
    >>> results.concordance("token", window=5)
    """

    def __init__(
        self,
        lf: pl.LazyFrame,
        query: str,
        matches: pl.DataFrame,
        variables: list[str],
        file_id_column: str,
        files: pl.DataFrame,
    ) -> None:
        self._lf = lf
        self._query = query
        self._matches = matches
        self._variables = variables
        self._file_id_column = file_id_column
        self._files = files

    def __len__(self) -> int:
        """Number of matches found."""
        return self._matches.height

    def _frame(self) -> pl.LazyFrame:
        return self._lf

    def _corpus_size(self) -> int:
        return int(self._files["_len"].sum())

    def _concordance(
        self,
        expr: IntoExprColumn | list[IntoExprColumn],
        window: int,
        chunk_column: Optional[str],
        metadata: Optional[list[str]],
        names: list[str],
    ) -> pl.DataFrame:
        mf = self._matches.with_row_index("_order").join(
            self._files.select("_file", "_offset", "_len", "_chunk"),
            on="_file",
            how="left",
        )
        columns = _needed_columns(expr, metadata, chunk_column, self._file_id_column)

        parts: list[pl.DataFrame] = []
        orders: list[int] = []
        for _, group in mf.group_by("_chunk", maintain_order=True):
            # Only out to the files the chunk has matches in, which is no more
            # than the chunk and often much less.
            offset, end = group.select(
                pl.col("_offset").min().alias("start"),
                (pl.col("_offset") + pl.col("_len")).max().alias("end"),
            ).row(0)
            chunk_lf = self._lf.slice(offset, end - offset)
            if columns is not None:
                chunk_lf = chunk_lf.select(columns)
            chunk_df = chunk_lf.collect(engine="streaming")
            matches = _absolute_matches(group, offset)
            file_ids = chunk_df.get_column(self._file_id_column)
            parts.append(
                _build_conc(
                    chunk_df,
                    matches,
                    expr,
                    window,
                    chunk_column,
                    metadata,
                    names,
                    file_ids,
                )
            )
            orders.extend(group["_order"].to_list())

        return (
            pl.concat(parts)
            .with_columns(pl.Series("_order", orders, dtype=pl.UInt32))
            .sort("_order")
            .drop("_order")
        )

    def _take(self, indices: Sequence[int]) -> LazySearchResults:
        return LazySearchResults(
            self._lf,
            self._query,
            self._matches[list(indices)],
            self._variables,
            self._file_id_column,
            self._files,
        )

    def with_spans_as_chunks(self, name: str = "spans") -> pl.LazyFrame:
        """
        Write the matches back onto the corpus as BIO tags, lazily.

        The lazy counterpart of `SearchResults.with_spans_as_chunks`. It tags
        the corpus the same way: "B" on the first token of each match, "I" on
        the rest, "O" elsewhere. Returning a LazyFrame lets the corpus be
        filtered down to the matched tokens before anything is materialized.

        Parameters
        ----------
        name : str, default "spans"
            Name for the tag column added to the corpus.

        Returns
        -------
        pl.LazyFrame
            The corpus, one row per token as before, with the tag column
            added.

        Examples
        --------
        >>> # Just the matched tokens, without materializing the corpus:
        >>> results.with_spans_as_chunks().filter(pl.col("spans") != "O").collect()
        """
        fid = self._file_id_column
        # File ids can be dictionary-encoded, and the tag frame's ids come
        # from another frame's dictionary; strings join reliably.
        key = pl.col(fid).cast(pl.String).alias("_fid")
        tags = (
            self._matches.lazy()
            .select(key, "start", pl.int_ranges("start", "end").alias("_pos"))
            # Matches are never zero-width, so there are no empty ranges to keep.
            .explode("_pos", empty_as_null=False)
            .select(
                "_fid",
                pl.col("_pos").cast(pl.Int64),
                pl.when(pl.col("_pos") == pl.col("start"))
                .then(pl.lit("B"))
                .otherwise(pl.lit("I"))
                .alias(name),
            )
        )
        return (
            self._lf.drop(name, strict=False)
            .with_columns(key, pl.int_range(pl.len()).over(fid).alias("_pos"))
            .join(tags, on=["_fid", "_pos"], how="left")
            .with_columns(pl.col(name).fill_null("O"))
            .drop("_fid", "_pos")
        )


def _absolute_matches(group: pl.DataFrame, chunk_offset: int) -> list[Match]:
    """The group's file-relative matches as absolute Match objects for a chunk.

    `group` carries `_offset`, each match's file's global offset; the chunk
    starts at `chunk_offset`, so a file's tokens sit at `_offset -
    chunk_offset` within the chunk.
    """
    bindings = (
        group["bindings"].to_list()
        if "bindings" in group.columns
        else [None] * group.height
    )
    matches = []
    for start, end, offset, bound in zip(
        group["start"], group["end"], group["_offset"], bindings
    ):
        base = int(offset) - chunk_offset
        spans = {
            name: Span(span["start"] + base, span["end"] + base)
            for name, span in (bound or {}).items()
            if span is not None
        }
        matches.append(Match(Span(start + base, end + base), spans))
    return matches


def concordance(
    search_results: SearchResults | LazySearchResults,
    expr: IntoExprColumn | list[IntoExprColumn] = "token",
    window: int = 0,
    chunk_column: Optional[str] = None,
    metadata: Optional[str | list[str]] = None,
    bindings: bool | str | list[str] = True,
) -> pl.DataFrame:
    """
    Lay the matches out as a KWIC (keyword in context) concordance.

    Parameters
    ----------
    search_results : SearchResults or LazySearchResults
        Matches to lay out, as `search` or `search_cqp` returned them.
    expr : IntoExprColumn or list of IntoExprColumn, default "token"
        Column name or expression to read the concordance text from (e.g.
        token or lemma). A list gives each its own trio of columns.
    window : int, default 0
        Words of context to take on each side of a match. The default of 0
        takes none, and the context columns are left off. Ignored when
        `chunk_column` is given.
    chunk_column : str, optional
        Column of BIO tags. Context then runs to the edges of the chunk
        holding the match instead of counting a fixed number of words. Pass a
        sentence tag column to get whole sentences.
    metadata : str or list of str, optional
        Column(s) of the corpus to carry through to each row as ordinary
        values rather than lists, e.g. `file_id` or `category`.
    bindings : bool, str or list of str, default True
        Which of the query's `$name:` captures to give a column of their own.
        True takes all of them, False none, and a name or list of names takes
        just those.

    Returns
    -------
    pl.DataFrame
        One row per match, in the order the results hold them, laid out as
        `SearchResults.concordance` describes.

    See Also
    --------
    SearchResults.concordance : Method interface for the same functionality.

    Examples
    --------
    >>> conc = concordance(results, "token", window=5)
    >>> conc = concordance(results, "token", chunk_column="sentence_tag")
    """
    return search_results.concordance(expr, window, chunk_column, metadata, bindings)


def collocates(
    search_results: SearchResults | LazySearchResults,
    expr: IntoExprColumn = "token",
    window: int = 5,
    min_freq: int = 5,
    freqs_name: str = "freqs",
) -> pl.DataFrame:
    """
    Count the words that occur near the matches.

    Parameters
    ----------
    search_results : SearchResults or LazySearchResults
        Matches to collocate around, as `search` or `search_cqp` returned them.
    expr : IntoExprColumn, default "token"
        Column name or expression the collocates are read from (e.g. token or
        lemma). It must name a single column.
    window : int, default 5
        Words to take on each side of each match. Must be at least 1.
    min_freq : int, default 5
        Times a word must occur in those windows to be reported. Pass 0 to
        keep them all.
    freqs_name : str, default "freqs"
        Name for the struct column the counts come back in.

    Returns
    -------
    pl.DataFrame
        One row per distinct word in the windows, with the four counts an
        association measure needs, laid out as `SearchResults.collocates`
        describes.

    See Also
    --------
    SearchResults.collocates : Method interface for the same functionality.

    Examples
    --------
    >>> collocs = collocates(results, "token", window=3)
    >>> collocs.with_columns(ll=pl.col("freqs").corpus.loglik())
    """
    return search_results.collocates(expr, window, min_freq, freqs_name)
