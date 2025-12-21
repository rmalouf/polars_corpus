"""Interactive concordance browser widget for Jupyter notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

import polars as pl

if TYPE_CHECKING:
    import anywidget

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

    Examples
    --------
    >>> results = plc.search(corpus, '[pos="ADJ"]')
    >>> conc = results.concordance('token', window=5)
    >>> widget = ConcordanceWidget(conc, 'token')
    >>> widget.show()
    """

    def __init__(
        self,
        df: pl.DataFrame,
        column: Optional[str] = None,
        page_size: int = 25,
    ) -> None:
        try:
            import anywidget
            import traitlets
        except ImportError:
            raise ImportError(
                "anywidget is required for ConcordanceWidget. "
                "Install with: pip install anywidget"
            )

        self.original_df = df
        self.df = df

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
        if self.column not in df.columns:
            raise ValueError(f"Missing required column: {self.column}")

        # Check if context columns exist
        self.has_context = self.left_col in df.columns and self.right_col in df.columns

        # Create the widget
        self.widget = self._create_widget(anywidget, traitlets, page_size)

    def _create_widget(
        self, anywidget: Any, traitlets: Any, page_size: int
    ) -> anywidget.AnyWidget:
        """Create the anywidget instance."""

        # Define widget class dynamically to avoid import-time dependency
        class _ConcordanceWidget(anywidget.AnyWidget):
            # Traits for state synchronization
            page_data = traitlets.List(trait=traitlets.Dict()).tag(sync=True)
            current_page = traitlets.Int(0).tag(sync=True)
            page_size = traitlets.Int(25).tag(sync=True)
            sort_column = traitlets.Unicode(None, allow_none=True).tag(sync=True)
            sort_descending = traitlets.Bool(False).tag(sync=True)
            filter_query = traitlets.Unicode("").tag(sync=True)
            view_mode = traitlets.Unicode("kwic").tag(sync=True)
            has_context = traitlets.Bool(True).tag(sync=True)
            column_name = traitlets.Unicode("").tag(sync=True)
            total_matches = traitlets.Int(0).tag(sync=True)
            show_advanced = traitlets.Bool(False).tag(sync=True)

            # Frontend JavaScript
            _esm = """
export function render({ model, el }) {
  // HTML escaping utility
  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // Debounce utility
  function debounce(func, wait) {
    let timeout;
    return function(...args) {
      clearTimeout(timeout);
      timeout = setTimeout(() => func.apply(this, args), wait);
    };
  }

  // Calculate derived values
  function getTotalPages() {
    const total = model.get('total_matches');
    const pageSize = model.get('page_size');
    return Math.max(1, Math.ceil(total / pageSize));
  }

  function getDisplayRange() {
    const currentPage = model.get('current_page');
    const pageSize = model.get('page_size');
    const total = model.get('total_matches');
    const start = currentPage * pageSize + 1;
    const end = Math.min((currentPage + 1) * pageSize, total);
    return { start, end, total };
  }

  // Build UI
  const container = document.createElement('div');
  container.className = 'concordance-widget';

  // Control bar
  const controlBar = document.createElement('div');
  controlBar.className = 'control-bar';

  const firstBtn = document.createElement('button');
  firstBtn.innerHTML = '⏮';
  firstBtn.title = 'First page';
  firstBtn.className = 'nav-btn';

  const prevBtn = document.createElement('button');
  prevBtn.innerHTML = '◀';
  prevBtn.title = 'Previous page';
  prevBtn.className = 'nav-btn';

  const pageInput = document.createElement('input');
  pageInput.type = 'number';
  pageInput.min = '1';
  pageInput.className = 'page-input';

  const nextBtn = document.createElement('button');
  nextBtn.innerHTML = '▶';
  nextBtn.title = 'Next page';
  nextBtn.className = 'nav-btn';

  const lastBtn = document.createElement('button');
  lastBtn.innerHTML = '⏭';
  lastBtn.title = 'Last page';
  lastBtn.className = 'nav-btn';

  const pageSizeSelect = document.createElement('select');
  pageSizeSelect.className = 'page-size-select';
  [10, 25, 50, 100, 200].forEach(size => {
    const option = document.createElement('option');
    option.value = size;
    option.textContent = size + ' per page';
    pageSizeSelect.appendChild(option);
  });

  const viewModeSelect = document.createElement('select');
  viewModeSelect.className = 'view-mode-select';
  [['KWIC', 'kwic'], ['Line', 'line']].forEach(([label, value]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    viewModeSelect.appendChild(option);
  });

  const shuffleBtn = document.createElement('button');
  shuffleBtn.innerHTML = '↻';
  shuffleBtn.title = 'Random shuffle';
  shuffleBtn.className = 'action-btn';

  const toggleBtn = document.createElement('button');
  toggleBtn.innerHTML = '⋮';
  toggleBtn.title = 'Show/hide sort & filter';
  toggleBtn.className = 'action-btn';

  controlBar.appendChild(firstBtn);
  controlBar.appendChild(prevBtn);
  controlBar.appendChild(pageInput);
  controlBar.appendChild(nextBtn);
  controlBar.appendChild(lastBtn);
  controlBar.appendChild(pageSizeSelect);
  controlBar.appendChild(viewModeSelect);
  controlBar.appendChild(shuffleBtn);
  controlBar.appendChild(toggleBtn);

  // Info label
  const infoLabel = document.createElement('div');
  infoLabel.className = 'info-label';

  // Advanced controls
  const advancedControls = document.createElement('div');
  advancedControls.className = 'advanced-controls';
  advancedControls.style.display = 'none';

  const sortRow = document.createElement('div');
  sortRow.className = 'control-row';

  const sortSelect = document.createElement('select');
  sortSelect.className = 'sort-select';
  const sortOptions = [['None', '']];
  const hasContext = model.get('has_context');
  if (hasContext) {
    sortOptions.push(['Left context', 'left'], ['Match', 'match'], ['Right context', 'right']);
  } else {
    sortOptions.push(['Match', 'match']);
  }
  sortOptions.forEach(([label, value]) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    sortSelect.appendChild(option);
  });

  const sortDirBtn = document.createElement('button');
  sortDirBtn.innerHTML = '↓';
  sortDirBtn.title = 'Sort descending';
  sortDirBtn.className = 'sort-dir-btn';
  sortDirBtn.dataset.descending = 'false';

  sortRow.appendChild(document.createTextNode('Sort: '));
  sortRow.appendChild(sortSelect);
  sortRow.appendChild(sortDirBtn);

  const filterRow = document.createElement('div');
  filterRow.className = 'control-row';

  const filterInput = document.createElement('input');
  filterInput.type = 'text';
  filterInput.placeholder = 'Filter...';
  filterInput.className = 'filter-input';

  const clearFilterBtn = document.createElement('button');
  clearFilterBtn.innerHTML = '✕';
  clearFilterBtn.title = 'Clear filter';
  clearFilterBtn.className = 'action-btn';

  filterRow.appendChild(document.createTextNode('Filter: '));
  filterRow.appendChild(filterInput);
  filterRow.appendChild(clearFilterBtn);

  advancedControls.appendChild(sortRow);
  advancedControls.appendChild(filterRow);

  // Output area
  const outputArea = document.createElement('div');
  outputArea.className = 'output-area';

  // Assemble container
  container.appendChild(controlBar);
  container.appendChild(infoLabel);
  container.appendChild(advancedControls);
  container.appendChild(outputArea);
  el.appendChild(container);

  // Render functions
  function renderTable() {
    const data = model.get('page_data');
    const viewMode = model.get('view_mode');
    const hasContext = model.get('has_context');

    if (data.length === 0) {
      outputArea.innerHTML = '<p><i>No concordances to display</i></p>';
      return;
    }

    if (viewMode === 'line') {
      renderLineView(data, hasContext);
    } else {
      renderKwicView(data, hasContext);
    }
  }

  function renderKwicView(data, hasContext) {
    const rows = data.map(row => {
      if (hasContext) {
        const left = escapeHtml(row.left || '');
        const match = escapeHtml(row.match || '');
        const right = escapeHtml(row.right || '');
        return `<tr>
          <td class="concordance-left">${left}</td>
          <td class="concordance-match">${match}</td>
          <td class="concordance-right">${right}</td>
        </tr>`;
      } else {
        const match = escapeHtml(row.match || '');
        return `<tr><td class="concordance-match-only">${match}</td></tr>`;
      }
    }).join('');

    outputArea.innerHTML = `<table class="concordance-table">${rows}</table>`;
  }

  function renderLineView(data, hasContext) {
    const rows = data.map(row => {
      const parts = [];
      if (hasContext) {
        if (row.left) parts.push(escapeHtml(row.left));
        if (row.match) parts.push(`<b>${escapeHtml(row.match)}</b>`);
        if (row.right) parts.push(escapeHtml(row.right));
      } else {
        if (row.match) parts.push(`<b>${escapeHtml(row.match)}</b>`);
      }
      const line = parts.join(' ');
      return `<tr><td class="concordance-line">${line}</td></tr>`;
    }).join('');

    outputArea.innerHTML = `<table class="concordance-table">${rows}</table>`;
  }

  function updateControls() {
    const currentPage = model.get('current_page');
    const totalPages = getTotalPages();
    const { start, end, total } = getDisplayRange();
    const pageSize = model.get('page_size');
    const viewMode = model.get('view_mode');
    const sortColumn = model.get('sort_column');
    const sortDescending = model.get('sort_descending');
    const filterQuery = model.get('filter_query');
    const showAdvanced = model.get('show_advanced');

    // Update buttons
    firstBtn.disabled = currentPage === 0;
    prevBtn.disabled = currentPage === 0;
    nextBtn.disabled = currentPage >= totalPages - 1;
    lastBtn.disabled = currentPage >= totalPages - 1;

    // Update page input
    pageInput.value = currentPage + 1;
    pageInput.max = totalPages;

    // Update info label
    if (total === 0) {
      infoLabel.innerHTML = '<span>No matches</span>';
    } else {
      infoLabel.innerHTML = `<span>Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()}</span>`;
    }

    // Update controls
    pageSizeSelect.value = pageSize;
    viewModeSelect.value = viewMode;
    sortSelect.value = sortColumn || '';
    filterInput.value = filterQuery;

    // Update sort direction button
    if (sortDescending) {
      sortDirBtn.innerHTML = '↑';
      sortDirBtn.title = 'Sort ascending';
      sortDirBtn.dataset.descending = 'true';
    } else {
      sortDirBtn.innerHTML = '↓';
      sortDirBtn.title = 'Sort descending';
      sortDirBtn.dataset.descending = 'false';
    }

    // Update advanced controls visibility
    if (showAdvanced) {
      advancedControls.style.display = 'block';
      toggleBtn.innerHTML = '⋯';
    } else {
      advancedControls.style.display = 'none';
      toggleBtn.innerHTML = '⋮';
    }
  }

  // Event handlers
  firstBtn.onclick = () => {
    model.set('current_page', 0);
    model.save_changes();
  };

  prevBtn.onclick = () => {
    const page = model.get('current_page');
    if (page > 0) {
      model.set('current_page', page - 1);
      model.save_changes();
    }
  };

  nextBtn.onclick = () => {
    const page = model.get('current_page');
    const totalPages = getTotalPages();
    if (page < totalPages - 1) {
      model.set('current_page', page + 1);
      model.save_changes();
    }
  };

  lastBtn.onclick = () => {
    model.set('current_page', Math.max(0, getTotalPages() - 1));
    model.save_changes();
  };

  pageInput.onchange = () => {
    const page = parseInt(pageInput.value) - 1;
    const totalPages = getTotalPages();
    if (page >= 0 && page < totalPages) {
      model.set('current_page', page);
      model.save_changes();
    }
  };

  pageSizeSelect.onchange = () => {
    model.set('page_size', parseInt(pageSizeSelect.value));
    model.set('current_page', 0);
    model.save_changes();
  };

  viewModeSelect.onchange = () => {
    model.set('view_mode', viewModeSelect.value);
    model.save_changes();
  };

  shuffleBtn.onclick = () => {
    // Trigger shuffle by setting a special value
    // This is handled in Python by detecting the change
    model.set('sort_column', null);
    model.set('current_page', 0);
    model.save_changes();
    // Trigger shuffle via custom message
    model.send({ type: 'shuffle' });
  };

  toggleBtn.onclick = () => {
    model.set('show_advanced', !model.get('show_advanced'));
    model.save_changes();
  };

  sortSelect.onchange = () => {
    const value = sortSelect.value;
    model.set('sort_column', value || null);
    model.save_changes();
  };

  sortDirBtn.onclick = () => {
    model.set('sort_descending', !model.get('sort_descending'));
    model.save_changes();
  };

  filterInput.oninput = debounce(() => {
    model.set('filter_query', filterInput.value);
    model.save_changes();
  }, 300);

  clearFilterBtn.onclick = () => {
    model.set('filter_query', '');
    model.save_changes();
  };

  // Listen for custom messages from Python
  model.on('msg:custom', (msg) => {
    if (msg.type === 'shuffle_done') {
      // Shuffle completed, refresh display
      renderTable();
      updateControls();
    }
  });

  // React to trait changes from Python
  model.on('change:page_data', renderTable);
  model.on('change:view_mode', renderTable);
  model.on('change:current_page', updateControls);
  model.on('change:page_size', updateControls);
  model.on('change:total_matches', updateControls);
  model.on('change:sort_column', updateControls);
  model.on('change:sort_descending', updateControls);
  model.on('change:filter_query', updateControls);
  model.on('change:show_advanced', updateControls);

  // Initial render
  renderTable();
  updateControls();
}
"""

            # CSS styling
            _css = """
.concordance-widget {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
}

.control-bar {
  display: flex;
  gap: 4px;
  align-items: center;
  margin: 2px 0;
  padding: 4px;
}

.nav-btn, .action-btn, .sort-dir-btn {
  padding: 2px 6px;
  border: 1px solid var(--jp-border-color1, rgba(0,0,0,0.2));
  background: var(--jp-layout-color1, white);
  color: var(--jp-content-font-color1, black);
  cursor: pointer;
  font-size: 14px;
  border-radius: 3px;
}

.nav-btn:hover, .action-btn:hover, .sort-dir-btn:hover {
  background: var(--jp-layout-color2, rgba(0,0,0,0.05));
}

.nav-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.page-input {
  width: 60px;
  padding: 2px 4px;
  border: 1px solid var(--jp-border-color1, rgba(0,0,0,0.2));
  background: var(--jp-layout-color1, white);
  color: var(--jp-content-font-color1, black);
  border-radius: 3px;
}

.page-size-select, .view-mode-select, .sort-select {
  padding: 2px 4px;
  border: 1px solid var(--jp-border-color1, rgba(0,0,0,0.2));
  background: var(--jp-layout-color1, white);
  color: var(--jp-content-font-color1, black);
  border-radius: 3px;
}

.info-label {
  font-size: 12px;
  opacity: 0.6;
  margin: 4px 0;
  padding: 0 4px;
}

.advanced-controls {
  padding: 4px;
  margin: 4px 0;
}

.control-row {
  display: flex;
  gap: 4px;
  align-items: center;
  margin: 4px 0;
}

.filter-input {
  flex: 1;
  padding: 2px 4px;
  border: 1px solid var(--jp-border-color1, rgba(0,0,0,0.2));
  background: var(--jp-layout-color1, white);
  color: var(--jp-content-font-color1, black);
  border-radius: 3px;
}

.output-area {
  margin: 8px 0;
}

.concordance-table {
  border-collapse: collapse;
  width: 100%;
  margin: 0;
}

.concordance-table td {
  padding: 2px 4px;
  vertical-align: middle;
  border-bottom: 1px solid var(--jp-border-color1, rgba(0,0,0,0.1));
}

.concordance-table tr:hover {
  background-color: var(--jp-layout-color2, rgba(0,0,0,0.05));
}

.concordance-left {
  text-align: right;
  padding-right: 8px;
}

.concordance-match {
  font-weight: bold;
  padding-left: 0;
  padding-right: 8px;
  text-align: center;
}

.concordance-right {
  text-align: left;
  padding-left: 0;
}

.concordance-match-only {
  text-align: left;
  font-weight: bold;
}

.concordance-line {
  text-align: left;
}

@media (prefers-color-scheme: dark) {
  .nav-btn, .action-btn, .sort-dir-btn, .page-input,
  .page-size-select, .view-mode-select, .sort-select, .filter-input {
    border-color: rgba(255,255,255,0.2);
    background: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.9);
  }

  .concordance-table td {
    border-bottom: 1px solid rgba(255,255,255,0.1);
  }

  .concordance-table tr:hover {
    background-color: rgba(255,255,255,0.05);
  }
}
"""

        # Create widget instance
        widget = _ConcordanceWidget()

        # Set read-only traits
        widget.has_context = self.has_context
        widget.column_name = self.column
        widget.total_matches = len(self.df)
        widget.page_size = page_size

        # Store reference to parent for access to DataFrames
        widget._parent = self

        # Set up observers
        @traitlets.observe("current_page", "page_size")
        def _on_pagination_change(change: Any) -> None:
            widget.page_data = self._prepare_page_data(widget)

        @traitlets.observe("sort_column", "sort_descending")
        def _on_sort_change(change: Any) -> None:
            if widget.sort_column:
                self._apply_sort(widget.sort_column, widget.sort_descending)
            else:
                # Restore original order (after filter if active)
                if widget.filter_query:
                    self._apply_filter(widget.filter_query)
                else:
                    self.df = self.original_df.clone()
            widget.current_page = 0
            widget.total_matches = len(self.df)
            widget.page_data = self._prepare_page_data(widget)

        @traitlets.observe("filter_query")
        def _on_filter_change(change: Any) -> None:
            self._apply_filter(widget.filter_query)
            # Reapply sort if active
            if widget.sort_column:
                self._apply_sort(widget.sort_column, widget.sort_descending)
            widget.current_page = 0
            widget.total_matches = len(self.df)
            widget.page_data = self._prepare_page_data(widget)

        # Bind observers to widget
        widget.observe(_on_pagination_change, names=["current_page", "page_size"])
        widget.observe(_on_sort_change, names=["sort_column", "sort_descending"])
        widget.observe(_on_filter_change, names=["filter_query"])

        # Handle custom messages (for shuffle)
        def _on_custom_msg(
            widget_instance: Any, content: Dict[str, Any], buffers: List[Any]
        ) -> None:
            if content.get("type") == "shuffle":
                self.df = self.df.sample(fraction=1.0, shuffle=True)
                widget.sort_column = None
                widget.current_page = 0
                widget.page_data = self._prepare_page_data(widget)
                # Send confirmation back to JavaScript
                widget.send({"type": "shuffle_done"})

        widget.on_msg(_on_custom_msg)

        # Initialize page data
        widget.page_data = self._prepare_page_data(widget)

        return widget

    def _prepare_page_data(self, widget: Any) -> List[Dict[str, str]]:
        """Convert current page slice to list of dicts with joined strings."""
        start = widget.current_page * widget.page_size
        page_df = self.df.slice(start, widget.page_size)

        if self.has_context:
            return [
                {
                    "left": " ".join(row[self.left_col]),
                    "match": " ".join(row[self.column]),
                    "right": " ".join(row[self.right_col]),
                }
                for row in page_df.iter_rows(named=True)
            ]
        else:
            return [
                {"match": " ".join(row[self.column])}
                for row in page_df.iter_rows(named=True)
            ]

    def _apply_sort(self, sort_column: str, descending: bool) -> None:
        """Apply sorting to dataframe."""
        # Map UI column names to DataFrame column names
        col_map = {
            "left": self.left_col,
            "match": self.column,
            "right": self.right_col,
        }
        df_column = col_map.get(sort_column)
        if df_column is None:
            return

        # Convert list column to string for sorting
        # For left context, reverse the list to sort by end of context
        if df_column == self.left_col:
            sort_expr = (
                pl.col(df_column).list.reverse().list.join(" ").str.to_lowercase()
            )
        else:
            sort_expr = pl.col(df_column).list.join(" ").str.to_lowercase()

        self.df = self.df.sort(by=sort_expr, descending=descending)

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

    def show(self) -> None:
        """Display the widget."""
        from IPython.display import display

        display(self.widget)  # type: ignore[no-untyped-call]
