from __future__ import annotations

import random
from typing import Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._internal import Span, py_concordance, py_kwic, spans_to_chunks

__all__ = ["SearchResults", "concordance", "collocates"]


class SearchResults:
    """Results of a corpus search query.

    This class wraps the results of a corpus search and provides methods for
    generating concordances, extracting matches, and manipulating result sets.
    SearchResults objects are typically created by the search functions and
    provide a fluent interface for working with search matches.

    Parameters
    ----------
    df : pl.DataFrame
        The source corpus DataFrame containing the original data.
    query : str
        The CQP query string that generated these results.
    matched_spans : list[Span]
        List of Span objects representing the matched text positions.

    Attributes
    ----------
    _df : pl.DataFrame
        The source corpus DataFrame.
    _query : str
        The original search query.
    _matched_spans : list[Span]
        The matched text spans.

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
        matched_spans: list[Span],
    ) -> None:
        self._df = df
        self._query = query
        self._matched_spans = matched_spans

    def __repr__(self) -> str:
        if len(self._matched_spans) != 1:
            es = "es"
        else:
            es = ""
        return f"SearchResults<'{self._query}'; {len(self._matched_spans):,} match{es}>"

    def concordance(
        self,
        expr: IntoExprColumn,
        window: Optional[int] = None,
        chunk_tag: Optional[str] = None,
    ) -> pl.DataFrame:
        """Generate a KWIC (Key Word In Context) concordance DataFrame.

        Creates a concordance table showing the matched text with surrounding
        context. The concordance format includes left context, the matched
        keyword(s), and right context in separate columns.

        Parameters
        ----------
        expr : IntoExprColumn
            Column name or Polars expression specifying which column(s)
            to use for generating the concordance display.
        window : int, optional
            Number of tokens to include on both left and right sides of
            each match. If None, returns matches with no context (equivalent
            to window=0). When chunk_tag is specified, this parameter is ignored.
        chunk_tag : str, optional
            Column name defining chunk boundaries for context extraction.
            When specified, context extends to chunk boundaries marked by
            "B" (begin) and "I" (inside) tags, ignoring the window parameter.
            Context stops at the first non-"I" tag in each direction.

        Returns
        -------
        pl.DataFrame
            Concordance DataFrame with columns named according to the input
            expression(s). For each column in expr, creates:
            - "{column}_left_context": List of tokens before the match
            - "{column}": List of matched tokens
            - "{column}_right_context": List of tokens after the match
            Context columns are omitted when window=0 or no context available.

        Examples
        --------
        >>> results.concordance("token", window=3)
        >>> results.concordance("token")  # No context, matches only
        >>> results.concordance("token", chunk_tag="sentence_tags")  # Chunk boundaries
        >>> results.concordance(["token", "pos"], window=5)  # Multiple columns
        """

        if chunk_tag is not None:
            chunk_tag_column = self._df.get_column(chunk_tag)
            return py_concordance(
                self._df.select(expr),
                self._matched_spans,
                chunk_tag_column,
            )
        else:
            if window is None:
                left_window = 0
                right_window = 0
            else:
                left_window = window
                right_window = window
            return py_kwic(
                self._df.select(expr),
                self._matched_spans,
                left_window,
                right_window,
            )

    def view(
        self,
        expr: IntoExprColumn = "token",
        window: Optional[int] = 5,
        chunk_tag: Optional[str] = None,
        page_size: int = 25,
    ) -> None:
        """Display an interactive concordance viewer in Jupyter.

        Creates and displays an interactive viewer for browsing concordance
        results with KWIC formatting, pagination, sorting, and filtering.

        Parameters
        ----------
        expr : IntoExprColumn, default "token"
            Column name or Polars expression specifying which column(s)
            to use for generating the concordance display.
        window : int, optional, default 5
            Number of tokens to include on both left and right sides of
            each match. If None, returns matches with no context (equivalent
            to window=0). When chunk_tag is specified, this parameter is ignored.
        chunk_tag : str, optional
            Column name defining chunk boundaries for context extraction.
            When specified, context extends to chunk boundaries marked by
            "B" (begin) and "I" (inside) tags, ignoring the window parameter.
        page_size : int, default 25
            Number of concordance lines to display per page.

        Notes
        -----
        Requires ipywidgets to be installed.

        Examples
        --------
        >>> results.view()  # Use defaults
        >>> results.view("token", window=10, page_size=50)
        >>> results.view("token", chunk_tag="sent_tag")
        """
        from .view import ConcordanceWidget

        # Generate concordance
        conc = self.concordance(expr, window=window, chunk_tag=chunk_tag)

        # Determine the column name
        if isinstance(expr, str):
            column = expr
        else:
            # If expr is a list or complex expression, use first column that's not context
            candidates = [
                col
                for col in conc.columns
                if not col.endswith("_left_context")
                and not col.endswith("_right_context")
            ]
            column = candidates[0] if candidates else None

        # Create and display widget
        widget = ConcordanceWidget(conc, column=column, page_size=page_size)
        widget.show()

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

        Examples
        --------
        >>> first_10 = results.head(10)
        """
        if abs(n) > len(self._matched_spans):
            return self
        else:
            return SearchResults(self._df, self._query, self._matched_spans[:n])

    def tail(self, n: int) -> SearchResults:
        """Return the last n search results.

        Parameters
        ----------
        n : int
            Number of results to return from the end.
            Must be positive. If n exceeds the total number of matches,
            returns all matches.

        Returns
        -------
        SearchResults
            New SearchResults object containing the last n matches.

        Raises
        ------
        ValueError
            If n is negative or zero.

        Examples
        --------
        >>> last_10 = results.tail(10)
        """
        if n > len(self._matched_spans):
            return self
        elif n > 0:
            return SearchResults(self._df, self._query, self._matched_spans[-n:])
        else:
            raise ValueError

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
        state = random.getstate()
        random.seed(seed)
        if k < 0 or k > len(self._matched_spans):
            raise ValueError
        try:
            new_results = SearchResults(
                self._df, self._query, random.sample(self._matched_spans, k)
            )
        finally:
            random.setstate(state)
        return new_results

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

        Examples
        --------
        >>> shuffled = results.shuffle(seed=123)
        """
        if seed is not None:
            state = random.getstate()
            random.seed(seed)
        try:
            new_results = SearchResults(
                self._df,
                self._query,
                random.sample(self._matched_spans, len(self._matched_spans)),
            )
        finally:
            if seed is not None:
                random.setstate(state)
        return new_results

    def with_spans_as_chunks(self, name: str = "spans") -> pl.DataFrame | pl.LazyFrame:
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
        DataFrame | LazyFrame
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
        return self._df.with_columns(
            spans_to_chunks(self._matched_spans, self._df.height).alias(name)
        )


def concordance(
    search_results: SearchResults,
    expr: IntoExprColumn,
    window: Optional[int] = None,
    chunk_tag: Optional[str] = None,
) -> pl.DataFrame:
    """Generate a concordance from search results (functional interface).

    This function provides a functional interface to SearchResults.concordance().
    It's equivalent to calling search_results.concordance(expr, window, chunk_tag).

    Parameters
    ----------
    search_results : SearchResults
        The search results to generate concordance from.
    expr : IntoExprColumn
        Column name or Polars expression for concordance display.
    window : int, optional
        Number of tokens to include on both left and right sides of
        each match. If None, returns matches with no context (equivalent
        to window=0). When chunk_tag is specified, this parameter is ignored.
    chunk_tag : str, optional
        Column name defining chunk boundaries for context extraction.
        When specified, context extends to chunk boundaries marked by
        "B" (begin) and "I" (inside) tags, ignoring the window parameter.

    Returns
    -------
    pl.DataFrame
        KWIC concordance DataFrame.

    See Also
    --------
    SearchResults.concordance : Method interface for the same functionality.

    Examples
    --------
    >>> conc = concordance(results, "token", window=5)
    >>> conc = concordance(results, "token", chunk_tag="sentence_tags")
    """
    return search_results.concordance(expr, window, chunk_tag)


def collocates(
    search_results: SearchResults, column: str, window_size: int = 5
) -> pl.DataFrame:
    """Extract collocate frequency information from search results.

    Analyzes the tokens that appear near the search matches within a specified
    window and computes frequency statistics useful for collocation analysis.
    Returns a DataFrame with collocate frequencies and corpus frequencies.

    Parameters
    ----------
    search_results : SearchResults
        The search results to analyze for collocates.
    column : str
        Name of the column containing the tokens to analyze.
    window_size : int, default 5
        Size of the context window on each side of the match
        (total window = 2 * window_size).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - collocate: The collocating token
        - f12: Frequency of collocate within the search context windows
        - f1: Total frequency of collocate in the corpus
        - f2: Number of search matches (constant for all rows)
        - n: Total number of context positions analyzed

    Notes
    -----
    The returned frequencies can be used to compute association measures
    like PMI, log-likelihood ratios, or other collocation statistics.

    Examples
    --------
    >>> collocs = collocates(results, "token", window_size=3)
    >>> # Find top collocates by raw frequency
    >>> top_collocs = collocs.sort("f12", descending=True).head(20)
    """
    f1 = search_results._df.lazy().group_by(column).len(name="f1")
    conc = concordance(search_results, column, window_size)
    tbl = (
        conc.lazy()
        .select(
            collocate=pl.col(f"{column}_left_context")
            .list.concat(f"{column}_right_context")
            .explode()
        )
        .group_by("collocate")
        .len(name="f12")
        .join(f1, left_on="collocate", right_on="token", how="left")
        .with_columns(
            f2=pl.lit(conc.height), n=search_results._df.height * window_size * 2
        )
    )
    return tbl.collect()
