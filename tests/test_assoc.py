import polars as pl
import pytest
from polars_corpus import crosstab, welchs_t
from polars.polars import ColumnNotFoundError, ComputeError
from polars.testing import assert_frame_equal


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
    with pytest.raises(ColumnNotFoundError):
        crosstab(df, "x", "y")
    with pytest.raises(ColumnNotFoundError):
        crosstab(df.lazy(), "x", "y").collect()


def test_crosstab_null_values() -> None:
    df = pl.DataFrame({"x": ["A", "A", "B", None, "C"], "y": [1, None, 1, 2, 1]})

    result = crosstab(df, "x", "y")
    assert len(result.filter(pl.col("x").is_null() | pl.col("y").is_null())) == 0
    assert len(result) > 0

    assert_frame_equal(
        result, crosstab(df.lazy(), "x", "y").collect(), check_row_order=False
    )


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

    assert_frame_equal(
        result, crosstab(df.lazy(), "x", "y").collect(), check_row_order=False
    )

def test_t_test() -> None:
    df = pl.DataFrame({'x':[1, 2, 3], 'y':[4, 5, 6]})

    r1 = df.select(welchs_t('x', 'y', alt="twosided")).unnest('t_test')
    assert -3.6742346 == pytest.approx(r1[('stat')].item())
    assert 0.02131164 == pytest.approx(r1[('pval')].item())
    assert 4.0 == pytest.approx(r1[('df')].item())

    r2 = df.select(welchs_t('x', 'y', alt="less")).unnest('t_test')
    assert -3.6742346 == pytest.approx(r2[('stat')].item())
    assert 0.01065582 == pytest.approx(r2[('pval')].item())
    assert 4.0 == pytest.approx(r2[('df')].item())

    r3 = df.select(welchs_t('x', 'y', alt="greater")).unnest('t_test')
    assert -3.6742346 == pytest.approx(r3[('stat')].item())
    assert 0.98934417 == pytest.approx(r3[('pval')].item())
    assert 4.0 == pytest.approx(r3[('df')].item())

def test_t_test_errors() -> None:

    df1 = pl.DataFrame({'x':[1], 'y':[2]})
    e1 = df1.select(welchs_t('x', 'y')).unnest('t_test')
    assert e1[('stat')].item() is None
    assert e1[('pval')].item() is None
    assert e1[('df')].item() is None

    df2 = pl.DataFrame({'x':[1, 1], 'y':[2, 2]})
    e2 = df2.select(welchs_t('x', 'y')).unnest('t_test')
    assert e2[('stat')].item() is None
    assert e2[('pval')].item() is None
    assert e2[('df')].item() is None

    with pytest.raises(ValueError):
        e3 = df1.select(welchs_t('x', 'y', alt='xyz')).unnest('t_test')
