from __future__ import annotations

import random
from typing import Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._internal import py_concordance, Span

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
            es = 'es'
        else:
            es = ''
        return f"SearchResults<'{self._query}'; {len(self._matched_spans):,} match{es}>"

    def concordance(
        self, expr: IntoExprColumn, context: str | int | tuple[int, int]
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
        context : str or int or tuple[int, int]
            Context specification:
            - str: Column name defining chunk boundaries (e.g., "sentence_id")
            - int: Number of tokens to include on both sides
            - tuple: (left_tokens, right_tokens) for asymmetric context

        Returns
        -------
        pl.DataFrame
            Concordance DataFrame with columns for left context, keywords,
            right context, and metadata about each match.

        Examples
        --------
        >>> results.concordance("token", 3)
        >>> results.concordance("token", (2, 4))  # 2 left, 4 right
        >>> results.concordance("token", "sentence_id")  # sentence boundaries
        """

        if isinstance(context, str):
            chunk_tag = self._df.get_column(context)
            left_window, right_window = 0, 0
        else:
            chunk_tag = None
            if isinstance(context, int):
                left_window = context
                right_window = context
            elif isinstance(context, tuple):
                left_window, right_window = context
            else:
                raise ValueError

        return py_concordance(
            self._df.select(expr),
            self._matched_spans,
            False,
            left_window,
            right_window,
            chunk_tag,
        )

    def matches(self, expr: pl.Expr) -> pl.DataFrame:
        """Extract the matched tokens without context.

        Returns just the tokens that were matched by the search query,
        without any surrounding context.

        Parameters
        ----------
        expr : pl.Expr
            Polars expression specifying which column(s) to extract
            from the matched spans.

        Returns
        -------
        pl.DataFrame
            DataFrame containing only the matched tokens/spans.

        Examples
        --------
        >>> results.matches(pl.col("token"))
        >>> results.matches(pl.col(["token", "pos"]))
        """
        return py_concordance(
            self._df.select(expr), self._matched_spans, True, 0, 0, None
        )

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

        Notes
        -----
        The random state is preserved, so this method does not affect
        the global random state. This is equivalent to sampling all
        matches without replacement.

        Examples
        --------
        >>> shuffled = results.shuffle(seed=123)
        """
        state = random.getstate()
        random.seed(seed)
        try:
            new_results = SearchResults(
                self._df,
                self._query,
                random.sample(self._matched_spans, len(self._matched_spans)),
            )
        finally:
            random.setstate(state)
        return new_results


def concordance(
    search_results: SearchResults,
    expr: IntoExprColumn,
    context: str | int | tuple[int, int],
) -> pl.DataFrame:
    """Generate a concordance from search results (functional interface).

    This function provides a functional interface to SearchResults.concordance().
    It's equivalent to calling search_results.concordance(expr, context).

    Parameters
    ----------
    search_results : SearchResults
        The search results to generate concordance from.
    expr : IntoExprColumn
        Column name or Polars expression for concordance display.
    context : str or int or tuple[int, int]
        Context specification (see SearchResults.concordance for details).

    Returns
    -------
    pl.DataFrame
        KWIC concordance DataFrame.

    See Also
    --------
    SearchResults.concordance : Method interface for the same functionality.

    Examples
    --------
    >>> conc = concordance(results, "token", 5)
    """
    return search_results.concordance(expr, context)


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
