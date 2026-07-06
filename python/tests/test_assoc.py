import math

import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError
from polars.testing import assert_frame_equal
from polars_corpus import chisq, crosstab, loglik, smp, welchs_t


def _chisq_ref(f12: int, f1: int, f2: int, n: int, yates: bool = False) -> float:
    """Reference chi-squared from the generic sum-of-(O-E)^2/E formula."""
    obs = [f12, f1 - f12, f2 - f12, n - f1 - f2 + f12]
    exp = [
        f1 * f2 / n,
        f1 * (n - f2) / n,
        (n - f1) * f2 / n,
        (n - f1) * (n - f2) / n,
    ]
    c = 0.5 if yates else 0.0
    return sum((abs(o - e) - c) ** 2 / e for o, e in zip(obs, exp))


def test_crosstab_basic() -> None:
    df = pl.DataFrame({"x": ["A", "A", "B", "B", "C"], "y": [1, 2, 1, 2, 1]})
    result = crosstab(df, "x", "y")

    assert "x" in result.columns
    assert "y" in result.columns
    assert "freqs" in result.columns

    # Verify struct fields
    freqs_dtype = result.schema["freqs"]
    assert isinstance(freqs_dtype, pl.Struct)
    field_names = [f.name for f in freqs_dtype.fields]
    assert "f12" in field_names
    assert "f1" in field_names
    assert "f2" in field_names
    assert "n" in field_names


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

    # Access f12 from the freqs struct
    f12 = pl.col("freqs").struct.field("f12")

    row_a = result.filter(pl.col("x") == "A")
    row_b = result.filter(pl.col("x") == "B")
    row_c = result.filter(pl.col("x") == "C")

    assert row_a.filter(pl.col("y") == 1).select(f12).to_series().to_list() == [1]
    assert row_a.filter(pl.col("y") == 2).select(f12).to_series().to_list() == [1]

    assert row_b.filter(pl.col("y") == 1).select(f12).to_series().to_list() == [1]
    assert row_b.filter(pl.col("y") == 2).select(f12).to_series().to_list() == [1]

    assert row_c.filter(pl.col("y") == 1).select(f12).to_series().to_list() == [2]
    assert row_c.filter(pl.col("y") == 2).select(f12).to_series().to_list() == [1]

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
        _ = df1.select(welchs_t("x", "y", alt="xyz")).unnest("t_test")


def test_compute_loglik() -> None:
    """Test compute_loglik with various scenarios and data types."""
    # Test data with both i64 (default) and u32 types
    test_data = {
        "f12": [6, 20, 0],
        "f1": [12, 30, 10],
        "f2": [10, 25, 20],
        "n": [20, 50, 50],
    }

    # Test with both i64 and u32 data types
    for dtype in [pl.Int64, pl.UInt32]:
        table = pl.DataFrame(test_data).cast(dtype)
        result = table.with_columns(LL=loglik("f12", "f1", "f2", "n"))

        # Check structure
        assert "LL" in result.columns
        assert len(result) == 3  # Invalid row filtered out

        ll_values = result["LL"].to_list()

        # Perfect independence case (f12=6) should be close to 0
        independence_ll = result.filter(pl.col("f12") == 6)["LL"].item()
        assert abs(independence_ll) < 1e-10

        # Manual verification for known case (f12=20)
        # o11=20, o12=10, o21=5, o22=15; e11=15, e12=15, e21=10, e22=10
        # o11 > e11 (20 > 15), so loglik is positive
        known_ll = result.filter(pl.col("f12") == 20)["LL"].item()
        expected = 2 * (
            20 * math.log(20 / 15)
            + 10 * math.log(10 / 15)
            + 5 * math.log(5 / 10)
            + 15 * math.log(15 / 10)
        )
        assert abs(known_ll - expected) < 1e-10

        # All values should be finite (can be positive or negative)
        assert all(math.isfinite(v) for v in ll_values)


def test_smp() -> None:
    # word 1: f12=2, f1=3 -> reference freq = f1-f12 = 1; (2+1)/(1+1) = 1.5
    # word 2: f12=0, f1=5 -> reference freq = 5; (0+1)/(5+1) = 1/6
    df = pl.DataFrame({"f12": [2, 0], "f1": [3, 5], "f2": [4, 4], "n": [7, 7]})
    result = df.with_columns(SMP=smp("f12", "f1", "f2", "n", 1.0))
    assert result["SMP"].to_list() == pytest.approx([1.5, 1 / 6])


@pytest.mark.parametrize("yates", [False, True])
def test_chisq(yates: bool) -> None:
    # Same tables as test_compute_loglik; the f12=6 row is perfect independence.
    test_data = {
        "f12": [6, 20, 0],
        "f1": [12, 30, 10],
        "f2": [10, 25, 20],
        "n": [20, 50, 50],
    }
    for dtype in [pl.Int64, pl.UInt32]:
        table = pl.DataFrame(test_data).cast(dtype)
        result = table.with_columns(X2=chisq("f12", "f1", "f2", "n", yates=yates))

        expected = [
            _chisq_ref(row["f12"], row["f1"], row["f2"], row["n"], yates)
            for row in table.iter_rows(named=True)
        ]
        assert result["X2"].to_list() == pytest.approx(expected)

        # Perfect independence (f12=6) is exactly 0 without correction.
        if not yates:
            assert result.filter(pl.col("f12") == 6)["X2"].item() == pytest.approx(0.0)


def test_chisq_default_uncorrected() -> None:
    # Default must omit Yates' correction (matches loglik-style raw statistic).
    df = pl.DataFrame({"f12": [20], "f1": [30], "f2": [25], "n": [50]})
    default = df.select(chisq("f12", "f1", "f2", "n")).item()
    assert default == pytest.approx(_chisq_ref(20, 30, 25, 50, yates=False))
    assert default != pytest.approx(_chisq_ref(20, 30, 25, 50, yates=True))


# Tests for struct-based expression namespace API


def test_struct_assoc_measures() -> None:
    """Test association measures via pl.col('freqs').corpus.* on crosstab output."""
    df = pl.DataFrame(
        {"x": ["A", "A", "B", "B", "C", "C", "C"], "y": [1, 2, 1, 2, 1, 1, 2]}
    )
    ct = crosstab(df, "x", "y")
    result = ct.with_columns(
        pl.col("freqs").corpus.loglik().alias("ll"),
        pl.col("freqs").corpus.pmi().alias("pmi"),
        pl.col("freqs").corpus.minsens().alias("minsens"),
        pl.col("freqs").corpus.smp(1.0).alias("smp"),
        pl.col("freqs").corpus.chisq().alias("chisq"),
    )

    # Verify row C, y=1: f12=2, f1=3, f2=4, n=7
    row = result.filter((pl.col("x") == "C") & (pl.col("y") == 1))
    f12, f1, f2, n = 2, 3, 4, 7

    # PMI = log(f12 * n / (f1 * f2))
    expected_pmi = math.log(f12 * n / (f1 * f2))
    assert row["pmi"].item() == pytest.approx(expected_pmi)

    # minsens = min(f12/f1, f12/f2)
    expected_minsens = min(f12 / f1, f12 / f2)
    assert row["minsens"].item() == pytest.approx(expected_minsens)

    # smp = (f12 + k) / ((f1 - f12) + k)
    expected_smp = (f12 + 1.0) / ((f1 - f12) + 1.0)
    assert row["smp"].item() == pytest.approx(expected_smp)

    # chisq matches the generic sum-of-(O-E)^2/E reference
    assert row["chisq"].item() == pytest.approx(_chisq_ref(f12, f1, f2, n))
