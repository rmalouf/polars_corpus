"""Tests for the concordance browser widget.

The JavaScript half is not exercised here; these cover the Python side it
drives -- which column it shows, and what sorting and filtering do to the
frame behind it.
"""

import polars as pl
import pytest
from polars_corpus import Match, Span

from .helpers import search_results
from polars_corpus.view import ConcordanceWidget


@pytest.fixture
def results():
    df = pl.DataFrame(
        {
            "token": "the cat sat on the mat ( x ) . the dog sat".split(),
            "pos": "DT NN VB IN DT NN PU X PU PU DT NN VB".split(),
            "file_id": ["a"] * 6 + ["b"] * 7,
        }
    )
    return search_results(df, "", [Match(Span(1, 2), {}), Match(Span(11, 12), {})])


@pytest.fixture
def conc(results):
    return results.concordance("token", window=3)


class TestColumn:
    """Which column the widget shows"""

    def test_named(self, conc):
        assert ConcordanceWidget(conc, "token").column == "token"

    def test_detected(self, conc):
        assert ConcordanceWidget(conc).column == "token"

    def test_metadata_is_not_the_matched_column(self, results):
        """Metadata columns are scalars; the matched ones hold lists."""
        with_meta = results.concordance("token", window=3, metadata="file_id")

        assert ConcordanceWidget(with_meta).column == "token"

    def test_first_of_several_matched_columns(self, results):
        both = results.concordance(["token", "pos"], window=3)

        assert ConcordanceWidget(both).column == "token"

    def test_no_matched_column(self):
        with pytest.raises(ValueError, match="none of the columns"):
            ConcordanceWidget(pl.DataFrame({"file_id": ["a"]}))

    def test_missing_named_column(self, conc):
        with pytest.raises(ValueError, match="no column 'nope'"):
            ConcordanceWidget(conc, "nope")


class TestContext:
    def test_with_context(self, conc):
        widget = ConcordanceWidget(conc)

        assert widget.has_context
        assert widget.widget.page_data[0] == {
            "left": "the",
            "match": "cat",
            "right": "sat on the",
        }

    def test_without_context(self, results):
        widget = ConcordanceWidget(results.concordance("token"))

        assert not widget.has_context
        assert widget.widget.page_data[0] == {"match": "cat"}


class TestPaging:
    def test_page_size(self, conc):
        widget = ConcordanceWidget(conc, page_size=1)

        assert widget.widget.total_matches == 2
        assert len(widget.widget.page_data) == 1

    def test_second_page(self, conc):
        widget = ConcordanceWidget(conc, page_size=1)
        widget.widget.current_page = 1

        assert widget.widget.page_data[0]["match"] == "dog"


class TestFilter:
    """What is typed in the filter box is matched as itself, not as a regex"""

    def test_matches_the_context(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.filter_query = "sat on"

        assert widget.df.height == 1

    def test_is_case_insensitive(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.filter_query = "CAT"

        assert widget.df.height == 1

    @pytest.mark.parametrize("query", [")", ") . the", "."])
    def test_regex_metacharacters_are_literal(self, conc, query):
        widget = ConcordanceWidget(conc)
        widget.widget.filter_query = query

        # These all appear in the left context of "dog" and nowhere else. As a
        # regex the first two raise, and "." matches every row.
        assert widget.df.height == 1

    def test_clearing_restores_every_row(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.filter_query = "cat"
        widget.widget.filter_query = ""

        assert widget.df.height == conc.height


class TestSort:
    def test_by_match(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.sort_column = "match"

        assert widget.df["token"].to_list() == [["cat"], ["dog"]]

    def test_descending(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.sort_column = "match"
        widget.widget.sort_descending = True

        assert widget.df["token"].to_list() == [["dog"], ["cat"]]

    def test_left_context_sorts_from_the_match_outwards(self, results):
        widget = ConcordanceWidget(results.concordance("token", window=3))
        widget.widget.sort_column = "left"

        # "... the cat" before "... the dog" -- read right to left, they only
        # differ at the token next to the match.
        assert [row[-1] for row in widget.df["token_left_context"]] == ["the", "the"]

    def test_clearing_restores_the_original_order(self, conc):
        widget = ConcordanceWidget(conc)
        widget.widget.sort_column = "match"
        widget.widget.sort_descending = True
        widget.widget.sort_column = None

        assert widget.df.equals(conc)


class TestShuffle:
    def test_survives_an_active_sort(self, conc):
        """Clearing the sort must not put the shuffled rows back in order."""
        widget = ConcordanceWidget(conc)
        widget.widget.sort_column = "match"

        widget.widget._handle_custom_msg({"type": "shuffle"}, [])

        assert widget.widget.sort_column is None
        assert sorted(m[0] for m in widget.df["token"]) == ["cat", "dog"]
