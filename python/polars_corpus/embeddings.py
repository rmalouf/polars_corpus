from __future__ import annotations

import polars as pl
from sentence_transformers import SentenceTransformer

from ._typing import IntoExpr
from .utils import as_expr

__all__ = [
    "encode",
]


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
