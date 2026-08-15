from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import polars as pl
from polars._typing import IntoExprColumn

from ._typing import IntoExpr
from .matcher import search
from .utils import as_eager, as_expr, check_columns

if TYPE_CHECKING:
    # sentence-transformers is in the `embeddings` extra, and importing it here
    # would put torch behind every `import polars_corpus`. Only the annotations
    # need it, and those aren't evaluated.
    from sentence_transformers import SentenceTransformer

__all__ = ["encode", "centroid", "encode_terms"]


def encode(model: SentenceTransformer, in_expr: IntoExpr) -> pl.Expr:
    """Encode a text column into fixed-width sentence embeddings.

    Parameters
    ----------
    model : SentenceTransformer
        Model used to encode `in_expr`'s strings into vectors.
    in_expr : IntoExpr
        Column name or expression holding the text to encode.

    Returns
    -------
    pl.Expr
        Expression producing one L2-normalized embedding vector per input
        string, as an `Array(Float32, dim)` with `dim` the model's embedding
        dimension.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> from sentence_transformers import SentenceTransformer
    >>> model = SentenceTransformer("all-MiniLM-L6-v2")
    >>> corpus.with_columns(emb=plc.encode(model, "lemma"))
    """
    expr = as_expr(in_expr)
    dim = model.get_embedding_dimension()
    return expr.map_batches(
        lambda col: model.encode(col.to_list(), normalize_embeddings=True),
        return_dtype=pl.Array(pl.Float32, dim),
        is_elementwise=True,
    )


def centroid(in_expr: IntoExpr, normalize: bool = True) -> pl.Expr:
    """Average a column of embedding vectors into a single vector.

    Parameters
    ----------
    in_expr : IntoExpr
        Column name or expression holding the vectors, as an
        `Array(Float32, dim)` -- what `encode` produces.
    normalize : bool, default True
        Scale the result to unit length, so that centroids of differently
        sized groups stay comparable under cosine similarity. An all-zero
        centroid is left as it is.

    Returns
    -------
    pl.Expr
        Aggregating expression producing one vector named "centroid".

    Examples
    --------
    >>> import polars_corpus as plc
    >>> corpus.group_by("category").agg(plc.centroid("emb"))
    >>> corpus.select(plc.centroid("emb", normalize=False))
    """
    expr = as_expr(in_expr)
    # Arrays don't mean() elementwise, but a struct's fields each mean() on
    # their own, so widen to one column per dimension and gather them back up.
    c = pl.concat_arr(expr.arr.to_struct().struct.unnest().mean()).alias("centroid")
    if normalize:
        x = pl.element()
        norm = (x * x).sum().sqrt()
        denom = pl.when(norm != 0).then(norm).otherwise(1.0)
        c = c.arr.eval(x / denom)
    return c


def encode_terms(
    terms: pl.DataFrame | pl.Series | Sequence[str],
    corpus: pl.DataFrame,
    model: SentenceTransformer,
    expr: IntoExprColumn = "token",
    window: int = 5,
    chunk_column: Optional[str] = None,
    max_matches: int = 10_000,
    seed: Optional[int] = 619,
    term_column: str = "token",
    **kwargs,
) -> pl.DataFrame:
    """Give each term a single vector, averaged over its uses in a corpus.

    Runs each term as a simple query, encodes every match together with the
    context it appears in, and averages those into one vector, so that terms
    used in similar contexts get similar vectors.

    Parameters
    ----------
    terms : pl.DataFrame, pl.Series or sequence of str
        Terms to encode, as simple queries. A DataFrame keeps its other
        columns, e.g. the frequencies the terms came with.
    corpus : pl.DataFrame
        Corpus to search for each term. It must be eager, as `search` is.
    model : SentenceTransformer
        Model used to encode the matches.
    expr : IntoExprColumn, default "token"
        Column name or expression holding the text to encode, as in
        `SearchResults.encode`.
    window : int, default 5
        Number of tokens of context on both sides of each match.
    chunk_column : str, optional
        Column name defining chunk boundaries. When given, context extends to
        the chunk holding the match and `window` is ignored.
    max_matches : int, default 10000
        Number of matches to encode per term. Terms with more than this are
        sampled down to it, so that a common term costs no more than a rare
        one.
    seed : int, optional
        Random seed for that sampling. The default makes results repeatable;
        pass None to sample differently each run.
    term_column : str, default "token"
        Column of `terms` holding the queries.
    **kwargs
        Passed on to `search`, e.g. `lemma_column=` or `file_id_column=`.

    Returns
    -------
    pl.DataFrame
        `terms` with a "vector" column of `Array(Float32, dim)`, holding each
        term's unit-length centroid. Terms with no matches in the corpus get a
        null vector.

    Examples
    --------
    >>> import polars_corpus as plc
    >>> from sentence_transformers import SentenceTransformer
    >>> model = SentenceTransformer("all-MiniLM-L6-v2")
    >>> plc.encode_terms(["bank", "shore", "vault"], corpus, model)
    >>> top = corpus.corpus.frequencies("token").head(500)
    >>> plc.encode_terms(top, corpus, model, chunk_column="sentence_tag")
    """
    if isinstance(terms, pl.DataFrame):
        frame = terms
    elif isinstance(terms, (pl.Series, list, tuple)):
        frame = pl.DataFrame({term_column: terms})
    else:
        raise ValueError(
            "terms must be a DataFrame with a column of queries, or a Series "
            f"or list of them, got {type(terms).__name__}"
        )
    check_columns(frame, [term_column], name="term list", param="term_column")
    corpus = as_eager(corpus)
    if max_matches < 1:
        raise ValueError(f"max_matches must be a positive integer, got {max_matches}")

    def term_vector(term: str) -> Optional[pl.Series]:
        results = search(corpus, term, **kwargs)
        if results is None:
            return None
        if len(results) > max_matches:
            results = results.sample(max_matches, seed=seed)
        encoded = results.encode(model, expr, window=window, chunk_column=chunk_column)
        return encoded.select(centroid("vector")).item()

    dim = model.get_embedding_dimension()
    return frame.with_columns(
        pl.col(term_column)
        .map_elements(term_vector, return_dtype=pl.Array(pl.Float32, dim))
        .alias("vector")
    )
