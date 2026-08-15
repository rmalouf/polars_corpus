from __future__ import annotations

import random
from difflib import get_close_matches
from typing import TYPE_CHECKING, Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._internal import Match, py_concordance, py_kwic, spans_to_chunks
from .utils import as_eager, check_columns, output_name

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

__all__ = ["SearchResults", "concordance", "collocates"]


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


class SearchResults:
    """Results of a corpus search query.

    This class wraps the results of a corpus search and provides methods for
    generating concordances, extracting matches, and manipulating result sets.
    SearchResults objects are typically created by the search functions and
    provide a fluent interface for working with search matches.

    Parameters
    ----------
    df : pl.DataFrame
        The source corpus DataFrame containing the original data. It must be
        eager: the matches index into it by position, and every method here
        reads its rows.
    query : str
        The CQP query string that generated these results.
    matches : list[Match]
        List of Match objects representing the matched text positions and bindings.
    variables : list[str], optional
        Names the query bound, in the order to show them. Read off the matches
        when not given.

    Attributes
    ----------
    _df : pl.DataFrame
        The source corpus DataFrame.
    _query : str
        The original search query.
    _matches : list[Match]
        The matched text spans with bindings.

    Examples
    --------
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> corpus = pl.DataFrame({"token": ["The", "quick", "brown", "fox"]})
    >>> results = plc.search(corpus, '[token="quick"]')
    >>> print(results)
    SearchResults<'[token="quick"]'; 1 match>
    """

    def __init__(
        self,
        df: pl.DataFrame,
        query: str,
        matches: list[Match],
        variables: Optional[list[str]] = None,
    ) -> None:
        self._df = as_eager(df)
        self._query = query
        self._matches = matches
        self._variables = variables

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
        if self._variables is None:
            # Matches built by hand carry no query to read the order off, and
            # each one holds its bindings unordered, so name them alphabetically.
            self._variables = sorted(
                {name for m in self._matches for name in m.bindings}
            )
        return self._variables

    def __repr__(self) -> str:
        if len(self._matches) != 1:
            es = "es"
        else:
            es = ""
        return f"SearchResults<'{self._query}'; {len(self._matches):,} match{es}>"

    def __len__(self) -> int:
        """Number of matches found."""
        return len(self._matches)

    def concordance(
        self,
        expr: IntoExprColumn | list[IntoExprColumn] = "token",
        window: int = 0,
        chunk_column: Optional[str] = None,
        metadata: Optional[str | list[str]] = None,
        bindings: bool | str | list[str] = True,
    ) -> pl.DataFrame:
        """Generate a KWIC (Key Word In Context) concordance DataFrame.

        Creates a concordance table showing the matched text with surrounding
        context. The concordance format includes left context, the matched
        keyword(s), and right context in separate columns.

        Parameters
        ----------
        expr : IntoExprColumn, default "token"
            Column name or Polars expression specifying which column(s)
            to use for generating the concordance display. A list names
            several at once.
        window : int, default 0
            Number of tokens to include on both left and right sides of
            each match. The default of 0 gives the matches with no context.
            Ignored when chunk_column is given.
        chunk_column : str, optional
            Column name defining chunk boundaries for context extraction.
            When specified, context extends to chunk boundaries marked by
            "B" (begin) and "I" (inside) tags, ignoring the window parameter.
            Context stops at the first non-"I" tag in each direction.
        metadata : str or list[str], optional
            Column name(s) from the source corpus to attach to each match as
            plain (non-list) columns, e.g. file_id or category. The value is
            taken from the first token of each match.
        bindings : bool, str or list[str], default True
            Which of the query's `$name:` bindings to give a column of their
            own. True takes them all -- and so does nothing to a query that
            binds none -- while False takes none. A name, or a list of names,
            takes just those; see the `variables` attribute for what a query
            bound.

        Returns
        -------
        pl.DataFrame
            Concordance DataFrame with columns named according to the input
            expression(s). For each column in expr, creates:
            - "{column}_left_context": List of tokens before the match
            - "{column}": List of matched tokens
            - "{column}_right_context": List of tokens after the match
            - "{column}_{variable}": List of tokens bound to each variable
            Context columns are omitted when window=0.
            Each name in metadata adds a single scalar column with that name.

        Notes
        -----
        Context is taken from the corpus in the order it is held, so it will
        run from the end of one file into the start of the next even when the
        search was confined to files with `file_id_column`.

        A binding column holds an empty list where the variable matched no
        token, as an optional one can, and a null where the match went down a
        branch of the query that never bound it at all.

        Examples
        --------
        >>> results.concordance("token", window=3)
        >>> results.concordance("token")  # No context, matches only
        >>> results.concordance("token", chunk_column="sentence_tag")  # Chunk boundaries
        >>> results.concordance(["token", "pos"], window=5)  # Multiple columns
        >>> results.concordance("token", window=5, metadata=["file_id", "category"])

        >>> # A column per bound variable: token_adj and token_noun
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

        if metadata is None:
            metadata_df = None
        else:
            if isinstance(metadata, str):
                metadata = [metadata]
            check_columns(self._df, metadata, param="metadata")
            metadata_df = self._df.select(metadata)

        if chunk_column is not None:
            check_columns(self._df, [chunk_column], param="chunk_column")
            chunk_tags = self._df.get_column(chunk_column)
            # The tags are read as strings. A dictionary-encoded column holds
            # the same ones and is worth the cast to get at them; a column of
            # anything else is not a tag column at all.
            if isinstance(chunk_tags.dtype, (pl.Categorical, pl.Enum)):
                chunk_tags = chunk_tags.cast(pl.String)
            elif chunk_tags.dtype != pl.String:
                raise ValueError(
                    f"chunk_column must hold the chunk tags 'B', 'I' and 'O' as "
                    f"strings, but {chunk_column!r} holds {chunk_tags.dtype}"
                )
            return py_concordance(
                _select(self._df, expr),
                self._matches,
                chunk_tags,
                metadata_df,
                names,
            )

        window = _check_count(window, "window", " Use window=0 for no context.")
        return py_kwic(
            _select(self._df, expr),
            self._matches,
            window,
            window,
            metadata_df,
            names,
        )

    def collocates(
        self,
        expr: IntoExprColumn = "token",
        window: int = 5,
        min_freq: int = 5,
        freqs_name: str = "freqs",
    ) -> pl.DataFrame:
        """Generate a collocate DataFrame.

        Analyzes the tokens that appear near the search matches within a specified
        window and computes frequency statistics useful for collocation analysis.
        Returns a DataFrame with collocate frequencies and corpus frequencies.

        Parameters
        ----------
        expr : IntoExprColumn, default "token"
            Column name or Polars expression specifying which column to use
            for collocate analysis. It must name a single column.
        window : int, default 5
            Number of tokens to include on both left and right sides of
            each match. Must be at least 1.
        min_freq : int, default 5
            Minimum frequency of each node-collocate pair within the window span.
            Node-collocate pairs with lower frequencies are not displayed in DataFrame.
            Pass 0 to keep them all.
        freqs_name : str, default "freqs"
            Name for the output frequencies struct column.

        Returns
        -------
        pl.DataFrame
            Collocate DataFrame with columns:
            - collocate: The collocating token
            - freqs: Struct with fields {f12, f1, f2, n} where:
                - f12: observed node-collocate frequency
                - f1: total number of window positions (matches * window * 2)
                - f2: total frequency of collocate in corpus
                - n: total words in corpus

        Notes
        -----
        The returned frequencies are laid out the way `crosstab` lays its own
        out, so the association measures take them as they come.

        `f1` counts the window positions the matches ask for, which is more
        than they get where a window runs off the end of the corpus.

        Examples
        --------
        >>> results.collocates("token")
        >>> results.collocates("token", window=3, min_freq=10)
        >>> collocs.with_columns(ll=pl.col("freqs").corpus.loglik())
        """
        if _check_count(window, "window") == 0:
            raise ValueError("window must be at least 1 to have any collocates in it")
        _check_count(min_freq, "min_freq", " Use min_freq=0 to keep them all.")
        # The concordance names its context columns after the expression's
        # output name, so the expression has to name one column, not several.
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
                # An empty context -- a match at the edge of the corpus -- has
                # no collocates in it rather than one null one.
                .explode(empty_as_null=False)
            )
            # A null token is not a collocate either.
            .drop_nulls()
            .group_by("collocate")
            .len(name="f12")
            .filter(pl.col("f12") >= min_freq)
            .join(
                self._df.lazy().group_by(expr).len(name="f2"),
                left_on="collocate",
                right_on=name,
                how="left",
            )
            .with_columns(n=self._df.height, f1=len(self._matches) * window * 2)
            .select("collocate", pl.struct("f12", "f1", "f2", "n").alias(freqs_name))
            .collect()
        )

    def view(
        self,
        expr: IntoExprColumn | list[IntoExprColumn] = "token",
        window: int = 5,
        chunk_column: Optional[str] = None,
        metadata: Optional[str | list[str]] = None,
        page_size: int = 25,
    ) -> None:
        """Display an interactive concordance viewer in Jupyter.

        Creates and displays an interactive viewer for browsing concordance
        results with KWIC formatting, pagination, sorting, and filtering.

        Parameters
        ----------
        expr : IntoExprColumn, default "token"
            Column name or Polars expression specifying which column(s)
            to use for generating the concordance display. Where it names
            several, the first is the one shown.
        window : int, default 5
            Number of tokens to include on both left and right sides of
            each match. Pass 0 for no context. Ignored when chunk_column
            is given.
        chunk_column : str, optional
            Column name defining chunk boundaries for context extraction.
            When specified, context extends to chunk boundaries marked by
            "B" (begin) and "I" (inside) tags, ignoring the window parameter.
        metadata : str or list[str], optional
            Column name(s) from the source corpus to attach to each match as
            plain (non-list) columns, e.g. file_id or category.
        page_size : int, default 25
            Number of concordance lines to display per page.

        Examples
        --------
        >>> results.view()  # Use defaults
        >>> results.view("token", window=10, page_size=50)
        >>> results.view("token", chunk_column="sent_tag")
        """
        from .view import ConcordanceWidget

        # The widget shows the matched column and its context, so the bound
        # variables would only take up room.
        conc = self.concordance(
            expr,
            window=window,
            chunk_column=chunk_column,
            metadata=metadata,
            bindings=False,
        )
        # The widget shows one column; the concordance holds the matched ones
        # ahead of any metadata, so its first is the one asked for.
        ConcordanceWidget(conc, page_size=page_size).show()

    def head(self, n: int) -> SearchResults:
        """Return the first n search results.

        Parameters
        ----------
        n : int
            Number of results to return from the beginning.
            If n exceeds the total number of matches, returns all matches.

        Returns
        -------
        SearchResults
            New SearchResults object containing the first n matches.

        Raises
        ------
        ValueError
            If n is negative.

        Examples
        --------
        >>> first_10 = results.head(10)
        """
        _check_count(n, "n")
        if n >= len(self._matches):
            return self
        return SearchResults(self._df, self._query, self._matches[:n], self._variables)

    def tail(self, n: int) -> SearchResults:
        """Return the last n search results.

        Parameters
        ----------
        n : int
            Number of results to return from the end.
            If n exceeds the total number of matches, returns all matches.

        Returns
        -------
        SearchResults
            New SearchResults object containing the last n matches.

        Raises
        ------
        ValueError
            If n is negative.

        Examples
        --------
        >>> last_10 = results.tail(10)
        """
        _check_count(n, "n")
        if n >= len(self._matches):
            return self
        # matches[-0:] is the whole list, not an empty one.
        return SearchResults(
            self._df,
            self._query,
            self._matches[len(self._matches) - n :],
            self._variables,
        )

    def sample(self, k: int, seed: Optional[int] = None) -> SearchResults:
        """Return a random sample of search results.

        Parameters
        ----------
        k : int
            Number of results to sample. Must be between 0 and the
            total number of matches.
        seed : int, optional
            Random seed for reproducible sampling. If None, uses
            the current random state.

        Returns
        -------
        SearchResults
            New SearchResults object containing k randomly sampled matches.

        Raises
        ------
        ValueError
            If k is negative or exceeds the number of available matches.

        Notes
        -----
        The random state is preserved, so this method does not affect
        the global random state.

        Examples
        --------
        >>> sample_100 = results.sample(100, seed=42)
        """
        _check_count(k, "k")
        if k > len(self._matches):
            raise ValueError(
                f"cannot sample {k:,} of {len(self._matches):,} matches; "
                f"k must be no larger than the number of matches"
            )
        state = random.getstate()
        random.seed(seed)
        try:
            return SearchResults(
                self._df,
                self._query,
                random.sample(self._matches, k),
                self._variables,
            )
        finally:
            random.setstate(state)

    # Do really want to do this? Am I assuming somewhere else that the spans are sorted?
    # We can always shuffle the concordance after it's built.
    def shuffle(self, seed: Optional[int] = None) -> SearchResults:
        """Return search results in randomized order.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible shuffling. If None, uses
            the current random state.

        Returns
        -------
        SearchResults
            New SearchResults object with matches in random order.

        Notes
        -----
        The random state is preserved, so this method does not affect
        the global random state.

        Examples
        --------
        >>> shuffled = results.shuffle(seed=123)
        """
        state = random.getstate()
        random.seed(seed)
        try:
            return SearchResults(
                self._df,
                self._query,
                random.sample(self._matches, len(self._matches)),
                self._variables,
            )
        finally:
            random.setstate(state)

    def with_spans_as_chunks(self, name: str = "spans") -> pl.DataFrame:
        """Add a column containing span information to the corpus DataFrame.

        Creates a new column in the corpus DataFrame that marks which tokens
        are part of search matches. The spans are represented as chunk tags
        where 'B' indicates the beginning of a match and 'I' indicates
        continuation of a match.

        Parameters
        ----------
        name : str, default 'spans'
            Name of the new column to add containing the span information.

        Returns
        -------
        pl.DataFrame
            The corpus DataFrame with the added spans column.

        Notes
        -----
        This method is useful for post-processing or visualization when you
        need to know which tokens in the corpus correspond to search matches.
        The span encoding follows the standard BIO (Begin-Inside-Outside)
        tagging scheme used in NLP.

        Examples
        --------
        >>> df_with_spans = results.with_spans_as_chunks('match_tags')
        >>> df_with_spans = results.with_spans_as_chunks()  # Default name 'spans'
        """
        # Extract spans from matches for Rust function
        spans = [match.span for match in self._matches]
        return self._df.with_columns(
            spans_to_chunks(spans, self._df.height).alias(name)
        )

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

        # The context columns are named after the expression's output name, so
        # the expression has to name one column, not several.
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
        # window=0 leaves the match with no context columns around it.
        parts = [
            column
            for column in (f"{name}_left_context", name, f"{name}_right_context")
            if column in conc.columns
        ]
        keep = [metadata] if isinstance(metadata, str) else list(metadata or [])
        return (
            conc.lazy()
            # Joining the lists rather than the joined strings keeps an empty
            # context -- a match at the edge of the corpus -- from padding the
            # line with a stray space.
            .select(pl.concat_list(parts).list.join(" ").alias(name), *keep)
            .with_columns(vector=encode(model, name))
            .collect()
        )


def concordance(
    search_results: SearchResults,
    expr: IntoExprColumn | list[IntoExprColumn] = "token",
    window: int = 0,
    chunk_column: Optional[str] = None,
    metadata: Optional[str | list[str]] = None,
    bindings: bool | str | list[str] = True,
) -> pl.DataFrame:
    """Generate a concordance from search results (functional interface).

    Parameters
    ----------
    search_results : SearchResults
        The search results to generate concordance from.
    expr : IntoExprColumn, default "token"
        Column name or Polars expression for concordance display. A list
        names several at once.
    window : int, default 0
        Number of tokens to include on both left and right sides of each
        match. The default of 0 gives the matches with no context. Ignored
        when chunk_column is given.
    chunk_column : str, optional
        Column name defining chunk boundaries for context extraction.
        When specified, context extends to chunk boundaries marked by
        "B" (begin) and "I" (inside) tags, ignoring the window parameter.
    metadata : str or list[str], optional
        Column name(s) from the source corpus to attach to each match as
        plain (non-list) columns, e.g. file_id or category.
    bindings : bool, str or list[str], default True
        Which of the query's `$name:` bindings to give a column of their own.
        True takes them all, False none, and a name or list of them just those.

    Returns
    -------
    pl.DataFrame
        KWIC concordance DataFrame, laid out as `SearchResults.concordance`
        describes.

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
    search_results: SearchResults,
    expr: IntoExprColumn = "token",
    window: int = 5,
    min_freq: int = 5,
    freqs_name: str = "freqs",
) -> pl.DataFrame:
    """Extract collocate frequency information from search results.

    Parameters
    ----------
    search_results : SearchResults
        The search results to analyze for collocates.
    expr : IntoExprColumn, default "token"
        Column name or Polars expression containing the tokens to analyze.
        It must name a single column.
    window : int, default 5
        Number of tokens to include on both left and right sides of each
        match. Must be at least 1.
    min_freq : int, default 5
        Minimum frequency of each node-collocate pair within the window span.
        Pass 0 to keep them all.
    freqs_name : str, default "freqs"
        Name for the output frequencies struct column.

    Returns
    -------
    pl.DataFrame
        Collocate DataFrame, laid out as `SearchResults.collocates` describes.

    See Also
    --------
    SearchResults.collocates : Method interface for the same functionality.

    Examples
    --------
    >>> collocs = collocates(results, "token", window=3)
    >>> collocs.with_columns(ll=pl.col("freqs").corpus.loglik())
    """
    return search_results.collocates(expr, window, min_freq, freqs_name)
