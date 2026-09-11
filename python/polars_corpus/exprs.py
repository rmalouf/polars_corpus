from __future__ import annotations

from typing import Any, Optional

import polars as pl

from ._typing import IntoExpr
from .assoc import (
    bic,
    chisq,
    crosstab,
    logdice,
    loglik,
    logratio,
    mi3,
    minsens,
    oddsratio,
    pctdiff,
    pmi,
    smp,
    tscore,
    zscore,
)
from .chunk import chunk_id, with_chunk_index
from .frequency import frequency_list
from .lexical import (
    count_hapaxes,
    frequency_spectrum,
    msttr,
    mtld,
    ttr,
    vocabulary_growth,
    yules_k,
)
from .matcher import search, search_cqp
from .search import LazySearchResults, SearchResults
from .utils import ngrams

# Which one comes back follows the frame searched; the namespace methods take
# either, so they say so.
_Results = SearchResults | LazySearchResults


@pl.api.register_expr_namespace("corpus")
class CorpusExpr:
    """
    The `.corpus` namespace on a Polars expression.

    The lexical-diversity measures read the tokens the expression gives.
    `chunk_id` reads a column of BIO tags. The association measures read a
    freqs struct with the fields `f12`, `f1`, `f2` and `n`, as
    `polars_corpus.crosstab` produces.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def ttr(self, **kwargs: Any) -> pl.Expr:
        """
        Calculate type-token ratio (TTR) of the tokens this expression gives.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        pl.Expr
            Expression returning the type-token ratio as a float scalar.

        See Also
        --------
        polars_corpus.ttr : Function form, with the details.
        """
        return ttr(self._expr, **kwargs)

    def msttr(self, **kwargs: Any) -> pl.Expr:
        """
        Calculate mean segmental type-token ratio (MSTTR) of the tokens this
        expression gives.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments for the function form, e.g. the segment
            size `n`.

        Returns
        -------
        pl.Expr
            Expression returning the mean segmental TTR as a float scalar,
            null if the text is too short to make one complete segment.

        See Also
        --------
        polars_corpus.msttr : Function form, with the details.
        """
        return msttr(self._expr, **kwargs)

    def yules_k(self, **kwargs: Any) -> pl.Expr:
        """
        Calculate Yule's K characteristic of the tokens this expression gives.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        pl.Expr
            Expression returning Yule's K as a float scalar, scaled by 10,000
            for readability.

        See Also
        --------
        polars_corpus.yules_k : Function form, with the details.
        """
        return yules_k(self._expr, **kwargs)

    def mtld(self, **kwargs: Any) -> pl.Expr:
        """
        Calculate Measure of Textual Lexical Diversity (MTLD) of the tokens
        this expression gives.

        Parameters
        ----------
        **kwargs
            Extra keyword arguments for the function form, e.g. the TTR
            `threshold`.

        Returns
        -------
        pl.Expr
            Expression returning the MTLD score as a float scalar, the mean
            number of tokens per factor.

        See Also
        --------
        polars_corpus.mtld : Function form, with the details.
        """
        return mtld(self._expr, **kwargs)

    def vocabulary_growth(self) -> pl.Expr:
        """
        Calculate a vocabulary growth curve V(N) over the tokens this
        expression gives.

        Returns
        -------
        pl.Expr
            Expression returning one type count per token, in corpus order.

        See Also
        --------
        polars_corpus.vocabulary_growth : Function form, with the details.
        """
        return vocabulary_growth(self._expr)

    def frequency_spectrum(self, sort: bool = False) -> pl.Expr:
        """
        Calculate the frequency spectrum of the tokens this expression gives.

        Parameters
        ----------
        sort : bool, default False
            Sort the result by frequency class. Unsorted, the rows come back
            in whatever order `value_counts` produced, which differs from one
            evaluation of the same expression to the next.

        Returns
        -------
        pl.Expr
            Expression returning one row per frequency class, as a struct
            with fields `m` (the frequency) and `V(m,N)` (the number of
            types that occur `m` times).

        See Also
        --------
        polars_corpus.frequency_spectrum : Function form, with the details.
        """
        return frequency_spectrum(self._expr, sort=sort)

    def count_hapaxes(self) -> pl.Expr:
        """
        Count the hapax legomena among the tokens this expression gives.

        Returns
        -------
        pl.Expr
            Expression returning the number of hapaxes as an integer scalar.

        See Also
        --------
        polars_corpus.count_hapaxes : Function form, with the details.
        """
        return count_hapaxes(self._expr)

    def ngrams(self, n: int) -> pl.Expr:
        """
        Gather each token together with the `n - 1` tokens that follow it.

        Parameters
        ----------
        n : int
            Length of the n-gram: 1 for unigrams, 2 for bigrams, and so on.

        Returns
        -------
        pl.Expr
            Expression giving the n-gram starting at each row. The last
            `n - 1` rows have no full n-gram to start and come out null.

        See Also
        --------
        polars_corpus.ngrams : Function form, with the details.
        """
        return ngrams(n, self._expr)

    def chunk_id(self) -> pl.Expr:
        """
        Number the chunks this column of BIO tags marks out.

        Returns
        -------
        pl.Expr
            Expression giving each token the number of "B" tags at or before
            it, so 1 for the first chunk and 2 for the second, and null where
            its tag is "O".

        See Also
        --------
        polars_corpus.chunk_id : Function form, with the details.

        Examples
        --------
        >>> df = pl.DataFrame({"bio": ["B", "I", "O", "B", "I"]})
        >>> df.with_columns(pl.col("bio").corpus.chunk_id().alias("chunk_idx"))
        shape: (5, 2)
        ┌─────┬───────────┐
        │ bio ┆ chunk_idx │
        │ --- ┆ ---       │
        │ str ┆ u32       │
        ╞═════╪═══════════╡
        │ B   ┆ 1         │
        │ I   ┆ 1         │
        │ O   ┆ null      │
        │ B   ┆ 2         │
        │ I   ┆ 2         │
        └─────┴───────────┘
        """
        return chunk_id(self._expr)

    def loglik(self) -> pl.Expr:
        """
        Compute the log-likelihood ratio (G²) from the freqs struct this
        expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the log-likelihood ratio for each row.

        See Also
        --------
        polars_corpus.loglik : Function form, with the details.
        """
        return loglik(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def pmi(self) -> pl.Expr:
        """
        Compute Pointwise Mutual Information (PMI) from the freqs struct this
        expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the PMI value for each row.

        See Also
        --------
        polars_corpus.pmi : Function form, with the details.
        """
        return pmi(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def mi3(self) -> pl.Expr:
        """
        Compute the MI3 association measure from the freqs struct this
        expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the MI3 value for each row.

        See Also
        --------
        polars_corpus.mi3 : Function form, with the details.
        """
        return mi3(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def logdice(self) -> pl.Expr:
        """
        Compute the log-Dice association measure from the freqs struct this
        expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the log-Dice value for each row.

        See Also
        --------
        polars_corpus.logdice : Function form, with the details.
        """
        return logdice(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def tscore(self) -> pl.Expr:
        """
        Compute the t-score from the freqs struct this expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the t-score for each row.

        See Also
        --------
        polars_corpus.tscore : Function form, with the details.
        """
        return tscore(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def zscore(self) -> pl.Expr:
        """
        Compute the z-score from the freqs struct this expression gives.

        Returns
        -------
        pl.Expr
            Expression computing the z-score for each row.

        See Also
        --------
        polars_corpus.zscore : Function form, with the details.
        """
        return zscore(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def minsens(self) -> pl.Expr:
        """
        Compute minimum sensitivity from the freqs struct this expression
        gives.

        Returns
        -------
        pl.Expr
            Expression computing the minimum sensitivity for each row.

        See Also
        --------
        polars_corpus.minsens : Function form, with the details.
        """
        return minsens(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def smp(self, k: float) -> pl.Expr:
        """
        Compute Kilgarriff's "simple maths" parameter from the freqs struct
        this expression gives.

        Parameters
        ----------
        k : float
            Smoothing constant added to both the target and reference
            frequencies.

        Returns
        -------
        pl.Expr
            Expression computing the simple maths value for each row.

        See Also
        --------
        polars_corpus.smp : Function form, with the details.
        """
        return smp(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            k,
        )

    def chisq(self, yates: bool = False) -> pl.Expr:
        """
        Compute Pearson's chi-squared (χ²) statistic from the freqs struct
        this expression gives.

        Parameters
        ----------
        yates : bool, default False
            Apply Yates' continuity correction for a 2×2 table.

        Returns
        -------
        pl.Expr
            Expression computing the chi-squared value for each row.

        See Also
        --------
        polars_corpus.chisq : Function form, with the details.
        """
        return chisq(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            yates,
        )

    def bic(self) -> pl.Expr:
        """
        Compute the Bayes factor BIC from the freqs struct this expression
        gives.

        Returns
        -------
        pl.Expr
            Expression computing the BIC value for each row.

        See Also
        --------
        polars_corpus.bic : Function form, with the details.
        """
        return bic(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
        )

    def logratio(self, discount: float = 0.5) -> pl.Expr:
        """
        Compute Hardie's log ratio from the freqs struct this expression
        gives.

        Parameters
        ----------
        discount : float, default 0.5
            Count to stand in for a frequency of zero, so that a word missing
            from one corpus gets a large log ratio rather than an infinite
            one.

        Returns
        -------
        pl.Expr
            Expression computing the log ratio for each row.

        See Also
        --------
        polars_corpus.logratio : Function form, with the details.
        """
        return logratio(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )

    def pctdiff(self, discount: float = 0.5) -> pl.Expr:
        """
        Compute %DIFF from the freqs struct this expression gives.

        Parameters
        ----------
        discount : float, default 0.5
            Count to stand in for a frequency of zero, so that a word missing
            from one corpus gets a large percentage rather than an infinite
            one.

        Returns
        -------
        pl.Expr
            Expression computing the %DIFF value for each row.

        See Also
        --------
        polars_corpus.pctdiff : Function form, with the details.
        """
        return pctdiff(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )

    def oddsratio(self, discount: float = 0.5) -> pl.Expr:
        """
        Compute the odds ratio from the freqs struct this expression gives.

        Parameters
        ----------
        discount : float, default 0.5
            Count to stand in for a cell of zero, so that a word missing from
            one corpus gets a large odds ratio rather than an infinite one.

        Returns
        -------
        pl.Expr
            Expression computing the odds ratio for each row.

        See Also
        --------
        polars_corpus.oddsratio : Function form, with the details.
        """
        return oddsratio(
            self._expr.struct.field("f12"),
            self._expr.struct.field("f1"),
            self._expr.struct.field("f2"),
            self._expr.struct.field("n"),
            discount,
        )


@pl.api.register_dataframe_namespace("corpus")
class CorpusDataFrame:
    """
    The `.corpus` namespace on a DataFrame.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def crosstab(self, x: str, y: str) -> pl.DataFrame:
        """
        Create a cross-tabulation (contingency table) from two categorical
        columns.

        Parameters
        ----------
        x : str
            Name of the column giving the first categorical variable.
        y : str
            Name of the column giving the second categorical variable.

        Returns
        -------
        pl.DataFrame
            One row per pair of levels, with a column for each of `x` and
            `y` named for the columns passed, and a `freqs` struct holding
            the joint frequency `f12`, the marginals `f1` and `f2`, and the
            grand total `n`.

        See Also
        --------
        polars_corpus.crosstab : Function form, with the details.
        """
        return crosstab(self._df, x, y)

    def frequency_list(self, expr: IntoExpr = "token", **kwargs: Any) -> pl.DataFrame:
        """
        Count how often each word occurs in the corpus.

        Parameters
        ----------
        expr : IntoExpr, default "token"
            Column name or expression identifying the word to count
            (e.g. token or lemma).
        **kwargs
            Extra keyword arguments for the function form, e.g. `basis`.

        Returns
        -------
        pl.DataFrame
            One row per type, most frequent first, with its `freq`, `rate`
            and `range`.

        See Also
        --------
        polars_corpus.frequency_list : Function form, with the details.
        """
        return frequency_list(self._df, expr, **kwargs)

    def with_chunk_index(self, chunk_column: str, **kwargs: Any) -> pl.DataFrame:
        """
        Add a column numbering the chunks that a column of BIO tags marks out.

        Parameters
        ----------
        chunk_column : str
            Column holding the BIO tags, e.g. a sentence tag.
        **kwargs
            Extra keyword arguments for the function form, e.g. `name` for
            the added column.

        Returns
        -------
        pl.DataFrame
            The frame with the chunk numbers added, named `chunk_idx` by
            default.

        See Also
        --------
        polars_corpus.with_chunk_index : Function form, with the details.
        """
        return with_chunk_index(self._df, chunk_column, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Optional[_Results]:
        """
        Find every place in a corpus where a simple (BNCweb-style) query
        matches.

        Parameters
        ----------
        query : str
            Simple query, e.g. `quick brown fox` or `{light}_V*`. See
            [Simple query language](simple_query.md) for the full syntax.
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        SearchResults or None
            The matches, or None if the query matched nothing.

        See Also
        --------
        polars_corpus.search : Function form, with the details.
        """
        return search(self._df, query, **kwargs)

    def search_cqp(self, query: str, **kwargs: Any) -> Optional[_Results]:
        """
        Find every place in a corpus where a CQP query matches.

        Parameters
        ----------
        query : str
            CQP query, e.g. `[pos="NN.*"] [lemma="be"]`. See
            [CQP query language](cqp_query.md) for the full syntax.
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        SearchResults or None
            The matches, or None if the query matched nothing.

        See Also
        --------
        polars_corpus.search_cqp : Function form, with the details.
        """
        return search_cqp(self._df, query, **kwargs)


@pl.api.register_lazyframe_namespace("corpus")
class CorpusLazyFrame:
    """
    The `.corpus` namespace on a LazyFrame.
    """

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def crosstab(self, x: str, y: str) -> pl.LazyFrame:
        """
        Create a cross-tabulation (contingency table) from two categorical
        columns.

        Parameters
        ----------
        x : str
            Name of the column giving the first categorical variable.
        y : str
            Name of the column giving the second categorical variable.

        Returns
        -------
        pl.LazyFrame
            One row per pair of levels, with a column for each of `x` and
            `y` named for the columns passed, and a `freqs` struct holding
            the joint frequency `f12`, the marginals `f1` and `f2`, and the
            grand total `n`.

        See Also
        --------
        polars_corpus.crosstab : Function form, with the details.
        """
        return crosstab(self._lf, x, y)

    def frequency_list(self, expr: IntoExpr = "token", **kwargs: Any) -> pl.LazyFrame:
        """
        Count how often each word occurs in the corpus.

        Parameters
        ----------
        expr : IntoExpr, default "token"
            Column name or expression identifying the word to count
            (e.g. token or lemma).
        **kwargs
            Extra keyword arguments for the function form, e.g. `basis`.

        Returns
        -------
        pl.LazyFrame
            One row per type, most frequent first, with its `freq`, `rate`
            and `range`.

        See Also
        --------
        polars_corpus.frequency_list : Function form, with the details.
        """
        return frequency_list(self._lf, expr, **kwargs)

    def with_chunk_index(self, chunk_column: str, **kwargs: Any) -> pl.LazyFrame:
        """
        Add a column numbering the chunks that a column of BIO tags marks out.

        Parameters
        ----------
        chunk_column : str
            Column holding the BIO tags, e.g. a sentence tag.
        **kwargs
            Extra keyword arguments for the function form, e.g. `name` for
            the added column.

        Returns
        -------
        pl.LazyFrame
            The frame with the chunk numbers added, named `chunk_idx` by
            default.

        See Also
        --------
        polars_corpus.with_chunk_index : Function form, with the details.
        """
        return with_chunk_index(self._lf, chunk_column, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Optional[_Results]:
        """
        Find every place in a corpus where a simple (BNCweb-style) query
        matches.

        Parameters
        ----------
        query : str
            Simple query, e.g. `quick brown fox` or `{light}_V*`. See
            [Simple query language](simple_query.md) for the full syntax.
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        LazySearchResults or None
            The matches, or None if the query matched nothing.

        See Also
        --------
        polars_corpus.search : Function form, with the details.
        """
        return search(self._lf, query, **kwargs)

    def search_cqp(self, query: str, **kwargs: Any) -> Optional[_Results]:
        """
        Find every place in a corpus where a CQP query matches.

        Parameters
        ----------
        query : str
            CQP query, e.g. `[pos="NN.*"] [lemma="be"]`. See
            [CQP query language](cqp_query.md) for the full syntax.
        **kwargs
            Extra keyword arguments for the function form.

        Returns
        -------
        LazySearchResults or None
            The matches, or None if the query matched nothing.

        See Also
        --------
        polars_corpus.search_cqp : Function form, with the details.
        """
        return search_cqp(self._lf, query, **kwargs)
