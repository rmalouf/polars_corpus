from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import polars as pl

if TYPE_CHECKING:
    import sys

    # import polars as pl

    if sys.version_info >= (3, 10):
        pass
    else:
        pass
    # from polars.datatypes import DataType, DataTypeClass

    # IntoExprColumn: TypeAlias = Union[pl.Expr, str, pl.Series]
    # PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]

T_Frame = TypeVar("T_Frame", pl.DataFrame, pl.LazyFrame)
