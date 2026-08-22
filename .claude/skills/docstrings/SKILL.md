---
name: docstrings
description: Write or revise a docstring in polars-corpus house style -- numpy sections, the public/internal register split, and what belongs in Parameters, Returns, Raises, Notes, and Examples. Use when adding a docstring to a new function or class, revising an existing one, or reviewing a change that adds public API.
---

# Writing docstrings

Numpy style throughout. mkdocstrings renders the public ones into `docs/*.md`
with `docstring_style: numpy` and `docstring_section_style: list`, so section
headings and their underlines must be exact or the section renders as prose.

The audience for public docstrings is a linguistics student who knows what they
want to measure and not much Polars. Write for someone reading the rendered
docs page, not someone reading the source.

## Two registers

**Public** -- named in the module's `__all__`, or a `.corpus` namespace method.
Summary starts on the line *after* the opening `"""`. Full sections. Every
parameter documented, including the `*_column` ones.

**Internal** -- helpers in `utils.py`, private functions, parser internals.
Summary on the *same* line as the opening `"""`. Then a short paragraph on why
the thing exists or what invariant it holds. Sections only when a caller
genuinely needs them; most internal helpers need none.

```python
# public
def dispersion(corpus, expr, method, min_freq=0, file_id_column="file_id"):
    """
    Measure how evenly each word is spread across the files of a corpus.

    Parameters
    ----------
    ...
    """

# internal
def check_expr(frame, expr, name="corpus", param="expr"):
    """Check that `frame` can evaluate `expr`, and name the column it produces.

    Resolving against the schema reads no data and settles what a column name
    alone cannot: regexes, selectors and positional references name their
    columns indirectly. `name` and `param` are used in error messages.
    """
```

`dispersion.py`, `keywords.py`, `lexical.py`, `assoc.py` and `visualizations.py`
are the current exemplars. `search.py`, `matcher.py` and `view.py` carry older,
more boilerplate-heavy docstrings -- do not copy their shape; bring them to this
style when you touch them.

## Summary line

One sentence, one line, ending in a period. Say what the function *measures* or
*produces*, in the reader's vocabulary:

- "Measure how evenly each word is spread across the files of a corpus."
- "Identify keywords by comparing frequencies in a target corpus against a reference corpus."
- "Plot the position of one or more words across a corpus as a barcode."

Not "This function computes...", not a restatement of the signature, not the
implementation. Slightly over 88 characters is acceptable to keep it one line.

An optional paragraph after the summary explains what the measure means and how
to read its values -- the thing a student needs before the parameter list is
useful. Skip it when the summary already says everything.

## Section order

`Parameters`, `Returns`, `Raises`, `Warns`, `Notes`, `References`, `See Also`,
`Examples`. Include a section only when it has something to say.

### Parameters

Type after ` : `, written for a reader, not copied from the annotation:

```
corpus : DataFrame | LazyFrame
expr : IntoExpr
method : {'chisq', 'll', 'minsens', 'pmi', 'smp', 'ttest'}
method : str | list of str
min_freq : int, default 0
ax : Axes, optional
**kwargs
```

- A fixed set of string choices goes in braces, alphabetical, or is spelled out
  as a bullet list in the body when each needs a gloss.
- `default X` for a value the reader would want to know; `optional` when the
  default is "not given" and the body explains what happens then.
- Backtick parameter names, column names, and code in prose: `` `min_freq` ``.
- `corpus`/`target`/`reference`: say what role the frame plays, not "the corpus".
- `expr`: "Column name or expression identifying the word/type to ... (e.g.
  token or lemma)." Note when it is evaluated against something other than the
  frame the reader passed.
- `*_column`: say what the column is *for*, not that it holds what its name
  says -- "Column holding file ids, defining the parts the word is spread
  across", not "Column name for file ids".
- Say when a parameter is ignored, required, or unused under some `method`.

### Returns

The type as rendered, then what the rows and columns actually are -- a reader
should know the shape of the result without running it.

- Frame-returning: name the columns and the ordering (or say "in no particular
  order"), and close with the eager/lazy rule: "Eager if `corpus` is a
  DataFrame, lazy if it is a LazyFrame."
- Expression-returning: `pl.Expr`, then "Expression returning ...".
- When one `method` returns a different shape than the others, give it its own
  paragraph.

### Raises

Group by exception type, one entry per type, listing the conditions in the
order the function checks them. For the standard public shape this mirrors the
`utils.py` guards:

```
ValueError
    If `corpus` is not a Polars DataFrame or LazyFrame, is empty, or is
    missing a column `dispersion` needs; if `expr` is not a column name or
    expression; or if `method` is not one of the measures listed above.
```

### Warns

Same shape as `Raises`, and say what the function does anyway: "It still gets a
row, drawn empty."

### Notes

Behavior a careful reader would otherwise have to discover: null handling
("Rows holding a null in either `expr` and `file_id_column` are dropped"),
stability caveats, why a default is what it is. Formulas go here, as
`$$ ... $$` blocks (katex via `pymdownx.arithmatex`). Use an `r"""` docstring
so the TeX needs no doubled backslashes.

### References

Bullet list, author-date, journal or book title in `*italics*`:

```
- Gries, S. Th. 2008. Dispersions and adjusted frequencies in corpora.
  *International Journal of Corpus Linguistics* 13(4): 403-437.
```

Cite the source of a measure, not general background.

### See Also

Only for a genuine sibling -- the method form of a function, or the measure a
reader would confuse this one with:

```
SearchResults.concordance : Method interface for the same functionality.
```

### Examples

`>>>` blocks, shortest real call first, then one or two that show what the
function is *for* -- a sort, a threshold, a comparison -- each with a `#`
comment saying why:

```
>>> import polars_corpus as plc
>>> plc.dispersion(corpus, "lemma", "d")
>>> # Rank the reasonably frequent words from least to most evenly spread:
>>> plc.dispersion(corpus, "lemma", "d", min_freq=50).sort("D")
```

Examples are illustrative, not executed. `corpus` and other setup may be left
undefined -- write the call a reader would type, not a self-contained script,
and do not add expected output to make an example runnable. Show output only
where a couple of lines genuinely clarify the result's shape.

Do not convert existing examples into runnable doctests. Making the examples
executable is deferred until the package is closer to shipping; until then,
leave `[tool.pytest.ini_options]` `testpaths` alone.

The one exception is already in place: `python/polars_corpus/simple_parser.py`
is on `testpaths` with `--doctest-modules`, because its examples are the
reference for the query language. Examples you add or edit *there* run under
`pytest` and need exact expected output.

## Cross-links

Markdown works in docstrings. Link to a docs page by filename:
`[Association metric](assoc.md)`. Pages live in `docs/`; check the target
exists before linking.

## Avoid

- Restating the signature ("Can be a column name (str) or Polars expression"
  repeated for every parameter -- say it once, in the `expr` entry).
- "This function ...", "Calculates ... by calculating ...".
- An `Attributes` section listing private attributes (`_df`, `_query`).
- Documenting internal helpers as if they were public API.
- Sections with nothing in them, or a `Notes` that repeats the summary.

## Check

After writing, confirm:

1. Section headings spelled exactly, underline the same length as the heading.
2. Every parameter in the signature appears, in signature order.
3. Frame-returning public functions state the eager/lazy rule.
4. `ruff format` leaves the file alone.
5. No example anywhere gained expected output for the sake of being runnable.
6. `pytest python/polars_corpus/simple_parser.py` still passes if you touched
   that file.
