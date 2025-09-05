from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, TypeAlias, Union

import polars as pl

if TYPE_CHECKING:
    import sys

    # import polars as pl

    if sys.version_info >= (3, 10):
        pass
    else:
        pass
    # from polars.datatypes import DataType, DataTypeClass
#    from polars.datatypes import Expr, Series
# PolarsDataType: TypeAlias = Union[DataType, DataTypeClass]

T_Frame = TypeVar("T_Frame", pl.DataFrame, pl.LazyFrame)

IntoExpr: TypeAlias = Union[pl.Expr, str]
