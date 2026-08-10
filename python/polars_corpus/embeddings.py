from __future__ import annotations

import polars as pl
from sentence_transformers import SentenceTransformer

from ._typing import IntoExpr
from .utils import as_expr

__all__ = ["encode", "centroid"]


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
