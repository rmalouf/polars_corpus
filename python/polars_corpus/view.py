"""Interactive concordance browser widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict

import polars as pl
from IPython.display import HTML, display

if TYPE_CHECKING:
    import ipywidgets as widgets


class WidgetChange(TypedDict, total=False):
    """Type for ipywidgets change event dictionaries."""

    new: Any
    old: Any
    owner: widgets.Widget
    name: str
    type: str


__all__ = ["ConcordanceWidget"]


class ConcordanceWidget:
    """Interactive browser for concordance DataFrames in Jupyter notebooks.

    Provides a linguist-friendly interface for navigating concordance results with
    KWIC (Key Word In Context) formatting, pagination, sorting, and filtering.

    Parameters
    ----------
    df : pl.DataFrame
        Concordance DataFrame with left context, match, and right context columns.
        Expected to have columns ending in '_left_context', the base column name,
        and '_right_context'.
    column : str, optional
        The base column name to display (e.g., 'token'). If None, attempts to
        detect from DataFrame columns.
    page_size : int, default 25
        Number of concordance lines to display per page.

    Attributes
    ----------
    df : pl.DataFrame
        The concordance DataFrame being displayed.
    column : str
        The column being displayed.
    current_page : int
        Current page number (0-indexed).
    page_size : int
        Number of lines per page.

    Examples
    --------
    >>> results = plc.search(corpus, '[pos="ADJ"]')
    >>> conc = results.concordance('token', window=5)
    >>> widget = ConcordanceWidget(conc, 'token')
    >>> widget.show()
    """

    # Widget instance attributes (initialized in _create_widgets)
    if TYPE_CHECKING:
        page_size_widget: widgets.Dropdown
        first_button: widgets.Button
        prev_button: widgets.Button
        next_button: widgets.Button
        last_button: widgets.Button
        page_input: widgets.BoundedIntText
        info_label: widgets.HTML
        sort_widget: widgets.Dropdown
        sort_direction_widget: widgets.ToggleButton
        filter_widget: widgets.Text
        clear_filter_button: widgets.Button
        sample_button: widgets.Button
        view_mode_widget: widgets.Dropdown
        output: widgets.Output
        advanced_controls: widgets.VBox
        toggle_button: widgets.Button
        container: widgets.VBox

    def __init__(
        self,
        df: pl.DataFrame,
        column: Optional[str] = None,
        page_size: int = 25,
    ) -> None:
        try:
            import ipywidgets as widgets
            from ipywidgets import Layout
        except ImportError:
            raise ImportError(
                "ipywidgets is required for ConcordanceWidget. "
                "Install with: pip install ipywidgets"
            )

        # Store for use in methods (needed to avoid import at module level)
        # These are stored as Any to avoid requiring ipywidgets at import time
        self._widgets_module: Any = widgets
        self._layout_class: Any = Layout

        self.df = df
        self.original_df = df  # Keep original for filtering

        # Auto-detect column if not provided
        if column is None:
            # Look for columns without _left_context or _right_context suffix
            candidates = [
                col
                for col in df.columns
                if not col.endswith("_left_context")
                and not col.endswith("_right_context")
            ]
            if candidates:
                column = candidates[0]
            else:
                raise ValueError("Could not auto-detect column name")

        self.column = column
        self.left_col = f"{column}_left_context"
        self.right_col = f"{column}_right_context"

        # Verify columns exist
        required_cols = [self.column]
        # left and right context are optional (for window=0)
        if self.left_col in df.columns:
            required_cols.append(self.left_col)
        if self.right_col in df.columns:
            required_cols.append(self.right_col)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.has_context = self.left_col in df.columns and self.right_col in df.columns

        self.page_size = page_size
        self.current_page = 0
        self.sort_column: Optional[str] = None
        self.sort_descending = False
        self.view_mode: Literal["kwic", "line"] = "kwic"

        # Create widgets
        self._create_widgets()

    def _create_widgets(self) -> None:
        """Create all widget components."""
        # Page size selector
        self.page_size_widget = self._widgets_module.Dropdown(
            options=[10, 25, 50, 100, 200],
            value=self.page_size,
            description="Per page:",
            layout=self._layout_class(width="120px"),
            style={"description_width": "55px"},
        )
        self.page_size_widget.observe(self._on_page_size_change, names="value")

        # Navigation buttons - compact and subtle
        self.first_button = self._widgets_module.Button(
            description="⏮",
            layout=self._layout_class(width="32px"),
            tooltip="First page",
        )
        self.first_button.on_click(self._on_first)

        self.prev_button = self._widgets_module.Button(
            description="◀",
            layout=self._layout_class(width="32px"),
            tooltip="Previous page",
        )
        self.prev_button.on_click(self._on_prev)

        self.next_button = self._widgets_module.Button(
            description="▶",
            layout=self._layout_class(width="32px"),
            tooltip="Next page",
        )
        self.next_button.on_click(self._on_next)

        self.last_button = self._widgets_module.Button(
            description="⏭",
            layout=self._layout_class(width="32px"),
            tooltip="Last page",
        )
        self.last_button.on_click(self._on_last)

        # Page number input
        self.page_input = self._widgets_module.BoundedIntText(
            value=1,
            min=1,
            max=max(1, self._total_pages()),
            description="Page:",
            layout=self._layout_class(width="100px"),
            style={"description_width": "35px"},
        )
        self.page_input.observe(self._on_page_input, names="value")

        # Info label - smaller font
        self.info_label = self._widgets_module.HTML(
            value=self._make_info_text(), layout=self._layout_class(margin="0 0 4px 0")
        )

        # Sort selector
        sort_options: list[tuple[str, Optional[str]]] = [("None", None)]
        if self.has_context:
            sort_options.extend(
                [
                    ("Left context", self.left_col),
                    ("Match", self.column),
                    ("Right context", self.right_col),
                ]
            )
        else:
            sort_options.append(("Match", self.column))

        self.sort_widget = self._widgets_module.Dropdown(
            options=sort_options,
            value=None,
            description="Sort:",
            layout=self._layout_class(width="160px"),
            style={"description_width": "32px"},
        )
        self.sort_widget.observe(self._on_sort_change, names="value")

        # Sort direction toggle
        self.sort_direction_widget = self._widgets_module.ToggleButton(
            value=False,
            description="↓",
            layout=self._layout_class(width="40px"),
            tooltip="Sort descending",
        )
        self.sort_direction_widget.observe(
            self._on_sort_direction_change, names="value"
        )

        # Filter box
        self.filter_widget = self._widgets_module.Text(
            placeholder="Filter...",
            description="Filter:",
            layout=self._layout_class(width="200px"),
            style={"description_width": "38px"},
        )
        self.filter_widget.observe(self._on_filter_change, names="value")

        # Clear filter button
        self.clear_filter_button = self._widgets_module.Button(
            description="✕",
            layout=self._layout_class(width="32px"),
            tooltip="Clear filter",
        )
        self.clear_filter_button.on_click(self._on_clear_filter)

        # Random sample button
        self.sample_button = self._widgets_module.Button(
            description="↻",
            layout=self._layout_class(width="32px"),
            tooltip="Random shuffle",
        )
        self.sample_button.on_click(self._on_random)

        # View mode selector
        self.view_mode_widget = self._widgets_module.Dropdown(
            options=[("KWIC", "kwic"), ("Line", "line")],
            value="kwic",
            description="View:",
            layout=self._layout_class(width="115px"),
            style={"description_width": "35px"},
        )
        self.view_mode_widget.observe(self._on_view_mode_change, names="value")

        # Output area for concordance display
        self.output = self._widgets_module.Output()

        # Collapsible advanced controls
        self.advanced_controls = self._widgets_module.VBox(
            [
                self._widgets_module.HBox(
                    [
                        self.sort_widget,
                        self.sort_direction_widget,
                    ]
                ),
                self._widgets_module.HBox(
                    [
                        self.filter_widget,
                        self.clear_filter_button,
                    ]
                ),
            ],
            layout=self._layout_class(margin="4px 0 0 0", display="none"),
        )

        # Toggle button for advanced controls
        self.toggle_button = self._widgets_module.Button(
            description="⋮",
            layout=self._layout_class(width="32px"),
            tooltip="Show/hide sort & filter",
        )
        self.toggle_button.on_click(self._on_toggle_advanced)

        # Layout - all controls in one compact row
        main_controls = self._widgets_module.HBox(
            [
                self.first_button,
                self.prev_button,
                self.page_input,
                self.next_button,
                self.last_button,
                self.page_size_widget,
                self.view_mode_widget,
                self.sample_button,
                self.toggle_button,
            ],
            layout=self._layout_class(margin="2px 0"),
        )

        controls_box = self._widgets_module.VBox(
            [
                main_controls,
                self.info_label,
                self.advanced_controls,
            ],
            layout=self._layout_class(padding="4px", margin="0 0 8px 0"),
        )

        self.container = self._widgets_module.VBox(
            [
                controls_box,
                self.output,
            ]
        )

    def _total_pages(self) -> int:
        """Calculate total number of pages."""
        total = len(self.df)
        return max(1, (total + self.page_size - 1) // self.page_size)

    def _make_info_text(self) -> str:
        """Generate info text showing current position."""
        total = len(self.df)
        if total == 0:
            return "<span style='font-size: 12px; opacity: 0.6;'>No matches</span>"

        start = self.current_page * self.page_size + 1
        end = min((self.current_page + 1) * self.page_size, total)

        total_display = f"{total:,}"
        if len(self.df) < len(self.original_df):
            total_display += f" of {len(self.original_df):,}"

        return f"<span style='font-size: 12px; opacity: 0.6;'>Showing {start:,}–{end:,} of {total_display}</span>"

    def _format_concordance_html(self) -> str:
        """Format current page as HTML with KWIC or Line layout."""
        start = self.current_page * self.page_size
        end = start + self.page_size
        page_df = self.df.slice(start, self.page_size)

        if len(page_df) == 0:
            return "<p><i>No concordances to display</i></p>"

        if self.view_mode == "line":
            return self._format_line_view(page_df)
        else:
            return self._format_kwic_view(page_df)

    def _format_kwic_view(self, page_df: pl.DataFrame) -> str:
        """Format as KWIC (3-column) view."""
        # Convert list columns to strings
        if self.has_context:
            display_df = page_df.select(
                [
                    pl.col(self.left_col).list.join(" ").alias("left"),
                    pl.col(self.column).list.join(" ").alias("match"),
                    pl.col(self.right_col).list.join(" ").alias("right"),
                ]
            )
        else:
            display_df = page_df.select(
                [
                    pl.col(self.column).list.join(" ").alias("match"),
                ]
            )

        # Build HTML table
        html = ["<style>"]
        html.append(
            '.concordance-table { border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; margin: 0; }'
        )
        html.append(
            ".concordance-table td { padding: 2px 4px; vertical-align: middle; border-bottom: 1px solid var(--jp-border-color1, rgba(0,0,0,0.1)); }"
        )
        html.append(
            ".concordance-table tr:hover { background-color: var(--jp-layout-color2, rgba(0,0,0,0.05)); }"
        )
        html.append(
            ".concordance-left { text-align: right !important; padding-right: 8px; }"
        )
        html.append(
            ".concordance-match { font-weight: bold; padding-left: 0; padding-right: 8px; text-align: center !important; }"
        )
        html.append(
            ".concordance-right { text-align: left !important; padding-left: 0; }"
        )
        html.append(
            ".concordance-match-only { text-align: left !important; font-weight: bold; }"
        )
        html.append("@media (prefers-color-scheme: dark) {")
        html.append(
            "  .concordance-table td { border-bottom: 1px solid rgba(255,255,255,0.1); }"
        )
        html.append(
            "  .concordance-table tr:hover { background-color: rgba(255,255,255,0.05); }"
        )
        html.append("}")
        html.append("</style>")

        html.append('<table class="concordance-table">')

        for row in display_df.iter_rows(named=True):
            html.append("<tr>")

            if self.has_context:
                # Escape HTML
                left = self._html_escape(row["left"])
                match = self._html_escape(row["match"])
                right = self._html_escape(row["right"])

                html.append(f'<td class="concordance-left">{left}</td>')
                html.append(f'<td class="concordance-match">{match}</td>')
                html.append(f'<td class="concordance-right">{right}</td>')
            else:
                match = self._html_escape(row["match"])
                html.append(f'<td class="concordance-match-only">{match}</td>')

            html.append("</tr>")

        html.append("</table>")

        return "\n".join(html)

    def _format_line_view(self, page_df: pl.DataFrame) -> str:
        """Format as line (single column) view with bold match."""
        # Build HTML table
        html = ["<style>"]
        html.append(
            '.concordance-table { border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 14px; margin: 0; width: 100%; }'
        )
        html.append(
            ".concordance-table td { padding: 2px 4px; vertical-align: middle; border-bottom: 1px solid var(--jp-border-color1, rgba(0,0,0,0.1)); text-align: left !important; }"
        )
        html.append(
            ".concordance-table tr:hover { background-color: var(--jp-layout-color2, rgba(0,0,0,0.05)); }"
        )
        html.append(".concordance-line { text-align: left !important; }")
        html.append("@media (prefers-color-scheme: dark) {")
        html.append(
            "  .concordance-table td { border-bottom: 1px solid rgba(255,255,255,0.1); }"
        )
        html.append(
            "  .concordance-table tr:hover { background-color: rgba(255,255,255,0.05); }"
        )
        html.append("}")
        html.append("</style>")

        html.append('<table class="concordance-table">')

        for row in page_df.iter_rows(named=True):
            html.append("<tr>")

            # Build line: left context + bold match + right context
            parts = []

            if self.has_context:
                left = row[self.left_col]
                if left:
                    parts.append(self._html_escape(" ".join(left)))

                match = row[self.column]
                if match:
                    parts.append(f"<b>{self._html_escape(' '.join(match))}</b>")

                right = row[self.right_col]
                if right:
                    parts.append(self._html_escape(" ".join(right)))
            else:
                match = row[self.column]
                if match:
                    parts.append(f"<b>{self._html_escape(' '.join(match))}</b>")

            line = " ".join(parts)
            html.append(f'<td class="concordance-line">{line}</td>')
            html.append("</tr>")

        html.append("</table>")

        return "\n".join(html)

    @staticmethod
    def _html_escape(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def _update_display(self) -> None:
        """Update the concordance display."""
        with self.output:
            self.output.clear_output(wait=True)
            display(HTML(self._format_concordance_html()))  # type: ignore[no-untyped-call]

        # Update controls
        self.info_label.value = self._make_info_text()
        self.page_input.max = max(1, self._total_pages())
        self.page_input.value = self.current_page + 1

        # Update button states
        total_pages = self._total_pages()
        self.first_button.disabled = self.current_page == 0
        self.prev_button.disabled = self.current_page == 0
        self.next_button.disabled = self.current_page >= total_pages - 1
        self.last_button.disabled = self.current_page >= total_pages - 1

    def _apply_sort(self) -> None:
        """Apply current sort settings to dataframe."""
        if self.sort_column is not None:
            # Convert list column to string for sorting
            # For left context, reverse the list to sort by end of context
            if self.sort_column == self.left_col:
                sort_expr = (
                    pl.col(self.sort_column)
                    .list.reverse()
                    .list.join(" ")
                    .str.to_lowercase()
                )
            else:
                sort_expr = pl.col(self.sort_column).list.join(" ").str.to_lowercase()
            self.df = self.df.sort(by=sort_expr, descending=self.sort_descending)

    def _apply_filter(self, query: str) -> None:
        """Apply filter to dataframe."""
        query = query.strip().lower()
        if not query:
            self.df = self.original_df.clone()
        else:
            # Filter: check if query appears in any column (case-insensitive)

            if self.has_context:
                mask = (
                    pl.col(self.left_col)
                    .list.join(" ")
                    .str.to_lowercase()
                    .str.contains(query)
                    | pl.col(self.column)
                    .list.join(" ")
                    .str.to_lowercase()
                    .str.contains(query)
                    | pl.col(self.right_col)
                    .list.join(" ")
                    .str.to_lowercase()
                    .str.contains(query)
                )
            else:
                mask = (
                    pl.col(self.column)
                    .list.join(" ")
                    .str.to_lowercase()
                    .str.contains(query)
                )

            self.df = self.original_df.filter(mask)

        # Reapply sort if active
        if self.sort_column is not None:
            self._apply_sort()

    # Event handlers
    def _on_page_size_change(self, change: WidgetChange) -> None:
        self.page_size = int(change["new"])
        self.current_page = 0
        self._update_display()

    def _on_first(self, button: widgets.Button) -> None:
        self.current_page = 0
        self._update_display()

    def _on_prev(self, button: widgets.Button) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self._update_display()

    def _on_next(self, button: widgets.Button) -> None:
        if self.current_page < self._total_pages() - 1:
            self.current_page += 1
            self._update_display()

    def _on_last(self, button: widgets.Button) -> None:
        self.current_page = max(0, self._total_pages() - 1)
        self._update_display()

    def _on_page_input(self, change: WidgetChange) -> None:
        page = int(change["new"]) - 1  # Convert to 0-indexed
        if 0 <= page < self._total_pages():
            self.current_page = page
            self._update_display()

    def _on_sort_change(self, change: WidgetChange) -> None:
        self.sort_column = change["new"]
        if self.sort_column is not None:
            self._apply_sort()
        else:
            # Restore original order
            self._apply_filter(self.filter_widget.value)
        self.current_page = 0
        self._update_display()

    def _on_sort_direction_change(self, change: WidgetChange) -> None:
        self.sort_descending = bool(change["new"])
        if change["new"]:
            self.sort_direction_widget.description = "↑"
            self.sort_direction_widget.tooltip = "Sort ascending"
        else:
            self.sort_direction_widget.description = "↓"
            self.sort_direction_widget.tooltip = "Sort descending"

        if self.sort_column is not None:
            self._apply_sort()
            self.current_page = 0
            self._update_display()

    def _on_filter_change(self, change: WidgetChange) -> None:
        self._apply_filter(str(change["new"]))
        self.current_page = 0
        self._update_display()

    def _on_clear_filter(self, button: widgets.Button) -> None:
        self.filter_widget.value = ""

    def _on_random(self, button: widgets.Button) -> None:
        self.df = self.df.sample(fraction=1.0, shuffle=True)
        self.sort_column = None
        self.sort_widget.value = None
        self.current_page = 0
        self._update_display()

    def _on_toggle_advanced(self, button: widgets.Button) -> None:
        if self.advanced_controls.layout.display == "none":
            self.advanced_controls.layout.display = "flex"
            self.toggle_button.description = "⋯"
        else:
            self.advanced_controls.layout.display = "none"
            self.toggle_button.description = "⋮"

    def _on_view_mode_change(self, change: WidgetChange) -> None:
        self.view_mode = str(change["new"])  # type: ignore[assignment]
        self._update_display()

    def show(self) -> None:
        """Display the widget."""
        self._update_display()
        display(self.container)  # type: ignore[no-untyped-call]
