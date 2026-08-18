import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from polars_corpus import chisq, crosstab, loglik, minsens, pmi, smp, welchs_t
from polars_corpus.assoc import welchs_t_from_stats
from polars_corpus.utils import output_name


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


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("a", "a"),
        (pl.col("a"), "a"),
        (pl.col("a").str.to_lowercase(), "a"),  # root name survives
        (pl.col("a").alias("b"), "b"),
        (pl.Series("a", ["x"]), "a"),  # a Series carries its own name
    ],
)
def test_output_name(expr, expected: str) -> None:
    assert output_name(expr) == expected


def test_crosstab_accepts_series() -> None:
    # crosstab evaluates its arguments against one frame, so a Series of
    # matching height is as good as a column name or an expression.
    df = pl.DataFrame({"x": ["A", "A", "B", "B", "C"], "y": [1, 2, 1, 2, 1]})
    series = pl.Series("x", df["x"].to_list())
    by_name = crosstab(df, "x", "y").sort("x", "y")
    by_series = crosstab(df, series, "y").sort("x", "y")
    assert_frame_equal(by_name, by_series)


@pytest.mark.parametrize("lazy", [False, True])
def test_crosstab_missing_columns(lazy: bool) -> None:
    # The missing column is reported up front, not out of a query plan, so a
    # LazyFrame raises at the call rather than at collect().
    df = pl.DataFrame({"a": ["A", "B"], "b": [1, 2]})
    with pytest.raises(ValueError, match="no column 'x'"):
        crosstab(df.lazy() if lazy else df, "x", "y")


def test_crosstab_drops_null_values() -> None:
    df = pl.DataFrame({"x": ["A", "A", "B", None, "C"], "y": [1, None, 1, 2, 1]})

    result = crosstab(df, "x", "y")
    assert len(result.filter(pl.col("x").is_null() | pl.col("y").is_null())) == 0
    assert len(result) > 0


def test_crosstab_correct_counts() -> None:
    df = pl.DataFrame(
        {"x": ["A", "A", "B", "B", "C", "C", "C"], "y": [1, 2, 1, 2, 1, 1, 2]}
    )
    result = crosstab(df, "x", "y").with_columns(
        f12=pl.col("freqs").struct.field("f12")
    )
    counts = {(x, y): f12 for x, y, f12 in result.select("x", "y", "f12").iter_rows()}

    assert counts == {
        ("A", 1): 1,
        ("A", 2): 1,
        ("B", 1): 1,
        ("B", 2): 1,
        ("C", 1): 2,
        ("C", 2): 1,
    }


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            {"x": ["A", "A", "B", "B", "C", "C", "C"], "y": [1, 2, 1, 2, 1, 1, 2]},
            id="complete",
        ),
        pytest.param(
            {"x": ["A", "A", "B", None, "C"], "y": [1, None, 1, 2, 1]}, id="with-nulls"
        ),
    ],
)
def test_crosstab_lazy_matches_eager(data: dict) -> None:
    df = pl.DataFrame(data)
    assert_frame_equal(
        crosstab(df, "x", "y"),
        crosstab(df.lazy(), "x", "y").collect(),
        check_row_order=False,
    )


@pytest.mark.parametrize(
    "alt,expected_pval",
    [("twosided", 0.02131164), ("less", 0.01065582), ("greater", 0.98934417)],
)
def test_t_test(alt: str, expected_pval: float) -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    result = df.select(welchs_t("x", "y", alt=alt)).unnest("t_test")

    assert result["t"].item() == pytest.approx(-3.6742346)
    assert result["p"].item() == pytest.approx(expected_pval)
    assert result["df"].item() == pytest.approx(4.0)
    # Hedges' g: Cohen's d of -3.0 (t * sqrt(1/n1 + 1/n2) with equal variances)
    # times the bias correction 1 - 3/(4*df - 1), the same for every alternative.
    assert result["g"].item() == pytest.approx(-2.4)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"x": [1], "y": [2]}, id="too-few-observations"),
        pytest.param({"x": [1, 1], "y": [2, 2]}, id="zero-variance"),
    ],
)
def test_t_test_undefined(data: dict) -> None:
    """Degenerate inputs yield nulls rather than raising."""
    result = pl.DataFrame(data).select(welchs_t("x", "y")).unnest("t_test")
    assert result.row(0) == (None, None, None, None)


def test_t_test_invalid_alternative() -> None:
    df = pl.DataFrame({"x": [1], "y": [2]})
    with pytest.raises(ValueError):
        df.select(welchs_t("x", "y", alt="xyz"))


# f12=6 is perfect independence; f12=20 is a known-value case; f12=0 is a zero cell.
CONTINGENCY = {
    "f12": [6, 20, 0],
    "f1": [12, 30, 10],
    "f2": [10, 25, 20],
    "n": [20, 50, 50],
}


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt32])
def test_loglik(dtype: pl.DataType) -> None:
    table = pl.DataFrame(CONTINGENCY).cast(dtype)
    result = table.with_columns(LL=loglik("f12", "f1", "f2", "n"))

    assert all(math.isfinite(v) for v in result["LL"])
    # Perfect independence is exactly zero.
    assert result.filter(pl.col("f12") == 6)["LL"].item() == pytest.approx(
        0.0, abs=1e-10
    )

    # f12=20: o11=20, o12=10, o21=5, o22=15; e11=15, e12=15, e21=10, e22=10.
    expected = 2 * (
        20 * math.log(20 / 15)
        + 10 * math.log(10 / 15)
        + 5 * math.log(5 / 10)
        + 15 * math.log(15 / 10)
    )
    assert result.filter(pl.col("f12") == 20)["LL"].item() == pytest.approx(expected)


def test_smp() -> None:
    # word 1: f12=2, f1=3 -> reference freq = f1-f12 = 1; (2+1)/(1+1) = 1.5
    # word 2: f12=0, f1=5 -> reference freq = 5; (0+1)/(5+1) = 1/6
    df = pl.DataFrame({"f12": [2, 0], "f1": [3, 5], "f2": [4, 4], "n": [7, 7]})
    result = df.with_columns(SMP=smp("f12", "f1", "f2", "n", 1.0))
    assert result["SMP"].to_list() == pytest.approx([1.5, 1 / 6])


@pytest.mark.parametrize("dtype", [pl.Int64, pl.UInt32])
@pytest.mark.parametrize("yates", [False, True])
def test_chisq(yates: bool, dtype: pl.DataType) -> None:
    table = pl.DataFrame(CONTINGENCY).cast(dtype)
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


# A well-behaved table, and each measure with the margins it actually reads: a
# null in one it ignores is no more its business than a column it never saw.
MARGINS = {"f12": 10, "f1": 100, "f2": 50, "n": 1000}
MEASURES = {
    "pmi": (pmi("f12", "f1", "f2", "n"), ("f12", "f1", "f2", "n")),
    "chisq": (chisq("f12", "f1", "f2", "n"), ("f12", "f1", "f2", "n")),
    "loglik": (loglik("f12", "f1", "f2", "n"), ("f12", "f1", "f2", "n")),
    "minsens": (minsens("f12", "f1", "f2", "n"), ("f12", "f1", "f2")),
    "smp": (smp("f12", "f1", "f2", "n", 1.0), ("f12", "f1")),
}


@pytest.mark.parametrize("measure, margins", MEASURES.values(), ids=MEASURES)
def test_measures_propagate_nulls(measure: pl.Expr, margins: tuple[str, ...]) -> None:
    """A null in a margin the measure reads makes it null, not a partial answer."""
    # One row per margin, holding a null there and sound counts everywhere else.
    table = pl.DataFrame(
        [MARGINS | {margin: None} for margin in margins],
        schema={name: pl.UInt64 for name in MARGINS},
    )
    values = table.select(measure).to_series()
    assert [m for m, value in zip(margins, values) if value is not None] == []


def test_measures_survive_unsigned_margins() -> None:
    """Margins that subtract below zero must not wrap around, as unsigned ints do."""
    # f12 > f1 is impossible from crosstab but reachable from a hand-built table.
    table = pl.DataFrame(
        [MARGINS | {"f12": 5, "f1": 0}], schema={name: pl.UInt64 for name in MARGINS}
    )
    measures = [measure.alias(name) for name, (measure, _) in MEASURES.items()]
    assert_frame_equal(table.select(measures), table.cast(pl.Int64).select(measures))


@pytest.mark.parametrize(
    "x, y, expected",
    [
        # Nulls leave the sample rather than counting toward its size:
        # scipy.stats.ttest_ind([1, 2, 3], [2, 4, 4, 5], equal_var=False)
        pytest.param(
            [1.0, 2.0, 3.0, None],
            [2.0, 4.0, 4.0, 5.0],
            (-2.04939015319192, 0.09648399932832219, 4.932885906040268),
            id="nulls-left-out",
        ),
        # A constant sample still leaves a usable standard error:
        # scipy.stats.ttest_ind(x, [0] * 6, equal_var=False)
        pytest.param(
            [0.10, 0.15, 0.08, 0.20, 0.12, 0.09],
            [0.0] * 6,
            (6.7106553135176075, 0.001112578255151147, 5.0),
            id="constant-sample",
        ),
    ],
)
def test_t_test_edge_cases(x: list, y: list, expected: tuple) -> None:
    result = pl.DataFrame({"x": x, "y": y}).select(welchs_t("x", "y")).unnest("t_test")
    assert result.row(0)[:3] == pytest.approx(expected)


def test_t_test_from_stats_propagates_nulls() -> None:
    """A null in any statistic, sample size included, gives a null result."""
    stats = {"s1": 6.0, "ss1": 14.0, "n1": 3.0, "s2": 9.0, "ss2": 29.0, "n2": 3.0}
    # One row per statistic, holding a null there and sound values elsewhere.
    table = pl.DataFrame([stats | {name: None} for name in stats])
    result = table.select(welchs_t_from_stats(*stats)).unnest("t_test")
    assert result.null_count().row(0) == (len(stats),) * 4
