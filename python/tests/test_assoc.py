import polars as pl
import pytest
from polars_corpus import crosstab, welchs_t, loglik
from polars.polars import ColumnNotFoundError, ComputeError
from polars.testing import assert_frame_equal
import math


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
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    r1 = df.select(welchs_t("x", "y", alt="twosided")).unnest("t_test")
    assert -3.6742346 == pytest.approx(r1[("stat")].item())
    assert pytest.approx(0.02131164) == pytest.approx(r1[("pval")].item())
    assert 4.0 == pytest.approx(r1[("df")].item())

    r2 = df.select(welchs_t("x", "y", alt="less")).unnest("t_test")
    assert -3.6742346 == pytest.approx(r2[("stat")].item())
    assert 0.01065582 == pytest.approx(r2[("pval")].item())
    assert 4.0 == pytest.approx(r2[("df")].item())

    r3 = df.select(welchs_t("x", "y", alt="greater")).unnest("t_test")
    assert -3.6742346 == pytest.approx(r3[("stat")].item())
    assert 0.98934417 == pytest.approx(r3[("pval")].item())
    assert 4.0 == pytest.approx(r3[("df")].item())


def test_t_test_errors() -> None:
    df1 = pl.DataFrame({"x": [1], "y": [2]})
    e1 = df1.select(welchs_t("x", "y")).unnest("t_test")
    assert e1[("stat")].item() is None
    assert e1[("pval")].item() is None
    assert e1[("df")].item() is None

    df2 = pl.DataFrame({"x": [1, 1], "y": [2, 2]})
    e2 = df2.select(welchs_t("x", "y")).unnest("t_test")
    assert e2[("stat")].item() is None
    assert e2[("pval")].item() is None
    assert e2[("df")].item() is None

    with pytest.raises(ValueError):
        e3 = df1.select(welchs_t("x", "y", alt="xyz")).unnest("t_test")


def test_compute_loglik() -> None:
    """Test compute_loglik with various scenarios and data types."""
    # Test data with both i64 (default) and u32 types
    test_data = {
        "f12": [6, 20, 0],
        "f1": [12, 30, 10],
        "f2": [10, 25, 20],
        "n": [20, 50, 50]
    }

    # Test with both i64 and u32 data types
    for dtype in [pl.Int64, pl.UInt32]:
        table = pl.DataFrame(test_data).cast(dtype)
        result = table.with_columns(LL=loglik("f12","f1","f2","n"))

        # Check structure
        assert "LL" in result.columns
        assert len(result) == 3  # Invalid row filtered out

        ll_values = result["LL"].to_list()

        # Perfect independence case (f12=6) should be close to 0
        independence_ll = result.filter(pl.col("f12") == 6)["LL"].item()
        assert abs(independence_ll) < 1e-10

        # Manual verification for known case (f12=20)
        # o11=20, o12=10, o21=5, o22=15; e11=15, e12=15, e21=10, e22=10
        known_ll = result.filter(pl.col("f12") == 20)["LL"].item()
        expected = 2 * (20*math.log(20/15) + 10*math.log(10/15) +
                       5*math.log(5/10) + 15*math.log(15/10))
        assert abs(known_ll - expected) < 1e-10

        # All values should be finite and non-negative
        assert all(math.isfinite(v) and v >= 0 for v in ll_values)
