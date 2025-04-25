import polars as pl
import pytest
from polars.polars import ComputeError
from polars_text.assoc import crosstab


def test_crosstab_basic() -> None:
    df = pl.DataFrame({"x": ["A", "A", "B", "B", "C"], "y": [1, 2, 1, 2, 1]})
    result = crosstab(df, "x", "y")

    assert "x" in result.columns
    assert "y" in result.columns
    assert "f12" in result.columns
    assert "f1" in result.columns
    assert "f2" in result.columns
    assert "n" in result.columns


def test_crosstab_missing_columns() -> None:
    df = pl.DataFrame({"a": ["A", "B"], "b": [1, 2]})
    with pytest.raises(ValueError, match="Columns x and/or y not found in dataframe"):
        crosstab(df, "x", "y")


def test_crosstab_null_values() -> None:
    df = pl.DataFrame({"x": ["A", "A", "B", None, "C"], "y": [1, None, 1, 2, 1]})
    result = crosstab(df, "x", "y")

    assert len(result.filter(pl.col("x").is_null() | pl.col("y").is_null())) == 0
    assert len(result) > 0


def test_crosstab_correct_counts() -> None:
    df = pl.DataFrame(
        {"x": ["A", "A", "B", "B", "C", "C", "C"], "y": [1, 2, 1, 2, 1, 1, 2]}
    )
    result = crosstab(df, "x", "y")

    row_a = result.filter(pl.col("x") == "A")
    row_b = result.filter(pl.col("x") == "B")
    row_c = result.filter(pl.col("x") == "C")

    assert row_a.filter(pl.col("y") == 1)["f12"].to_list() == [1]
    assert row_a.filter(pl.col("y") == 2)["f12"].to_list() == [1]

    assert row_b.filter(pl.col("y") == 1)["f12"].to_list() == [1]
    assert row_b.filter(pl.col("y") == 2)["f12"].to_list() == [1]

    assert row_c.filter(pl.col("y") == 1)["f12"].to_list() == [2]
    assert row_c.filter(pl.col("y") == 2)["f12"].to_list() == [1]
