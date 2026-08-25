---
name: example-notebook-annotation
description: Add explanatory markdown cells to one of the example notebooks in docs/notebooks/, which mkdocs renders into the documentation site. The prose carries the narrative layer of the docs -- how the library expresses an analysis and what its calls decide on the reader's behalf -- for a reader who already knows corpus linguistics. Use whenever the user mentions annotating, narrating, writing up, or "adding markdown to" an example notebook, or asks to make one ready for the docs site.
---

# Annotating an example notebook

The example notebooks are the narrative half of the documentation. `mkdocs-jupyter`
renders them under **Examples** in the nav, opposite the **Reference** pages, and the
three layers divide as follows:

- a reference page is a heading and a `:::` directive, nothing else;
- a docstring is the contract for one function, written for a student who does not
  know Polars;
- a notebook shows what a whole analysis looks like end to end, and is the only
  place a reader sees several calls composed into one pipeline.

So the notes carry what neither of the other two can: how the pieces fit, which of
the library's choices are load-bearing, and what a real result looks like. Do not
restate a docstring here, and do not use a notebook to document one function.

The reader knows corpus linguistics. They know what a type/token ratio, a
concordance, a keyword, a lemma, a KWIC line and a dispersion measure are, and why
someone would want one. They are here to learn how *this library* gets them, and
what it decided on their behalf along the way. Explaining the linguistics is the
main way these notes go wrong.

`frequencies.ipynb` and `keywords.ipynb` are the exemplars. Read at least one
before drafting.

## Process

1. **Read the whole notebook first**, outputs included. What a cell is for is often
   only visible from what it feeds three cells later.
2. **Read the source** for every library call the notebook makes, in
   `python/polars_corpus/`. Nearly every note makes a claim about behavior, and the
   notebook cannot be re-run to check one.
3. **Reconstruct the arc.** A notebook is three to six moves -- read and normalize
   the corpus, compute the measure, read the result, show what the measure misses,
   reach for a second one. Those seams become `##` headings.
4. **Work through the cells** and write the notes, most of which are short.
5. **Verify** that the code cells and their outputs are untouched, using the script
   below, and report back.

## Density

Most code cells should end up with a markdown cell in front of them, and the
notebook should have more annotated cells than unannotated ones. Length varies
widely. A cell at a seam may need a heading and a paragraph or two; a routine step
needs a clause, as in "the `min_freq=200` cutoff drops the words whose per-file
counts are too sparse for `D` to mean much." A very short note beats no note.

Skip a cell when the same move was explained earlier and this repeats it, when it
continues directly from the cell above and that note covers both, when it already
has a markdown cell, or when it is pure setup -- imports, `pl.Config` calls, a
path. Existing markdown stays exactly as it is and is not restated in a new cell.

## What the note contains

Depending on the cell, some combination of the following. The first two are where
almost all the value is.

**What the call does, and what its result looks like.** Which parameter selected
this behavior, what the columns of the returned frame are, what the ordering is,
what it drops. Write it the way `keywords.ipynb` does: "that is `method='ttest'`,
which returns only the words more frequent in the target, ranked by `p`
ascending." Name the column the reader is about to see in the output. Where a
docstring covers the rest, link the reference page rather than paraphrasing it --
but see the note on link form below, because a notebook cannot link the way a
markdown page does.

**The analytic choice, and what it costs.** Every pipeline decides things: case
folding, `is_letters` over a regex, which column plays the type role, a frequency
threshold, a window size, per-file rates over a corpus-wide one, lemma over form.
Name the decision and say what it renders invisible. The reader can supply the
linguistics; what they cannot supply is which line of code made the choice.

**Where the library ends and Polars begins.** A reader's first question is usually
how much of this they could have written themselves. Say when a step is ordinary
Polars -- `group_by`/`agg`, a selector, a window function -- and when it needs
something from `plc`, and why: the Rust matcher, an aggregation that has no
expression form, a measure that needs the whole file at once.

**A reading of the output.** Only for cells whose output has something in it to
notice, and always pointed at what is actually printed rather than described in
general terms.

**What will cost the reader something.** Corpora that live outside the repo,
a call that collects a lazy frame, memory that scales with the vocabulary rather
than the corpus, a plotting import from the `examples` extra, anything that would
behave differently on a corpus annotated with other column names.

Keep notes about the plotting thin. The plotting layer is expected to move off
seaborn, so prose built around a seaborn call is prose that will need rewriting;
say what the figure shows, not how the call was built.

## Register

Continuous prose, full sentences, the register of the reference works the notebooks
cite. Avoid the clipped declarative stack ("Prose. No bullets. No emoji."), which
belongs to marketing copy and sits badly next to code. A very short note can be a
fragment when there is only one thing to say.

First person plural, present tense -- "we read the BNC, restrict it to face-to-face
conversation" -- matching the notebooks already written. Begin with the substance,
not with "In this cell we", "Let's now", "As you can see", or "It's important to
note that." Nothing here needs praising, so leave out "elegant", "powerful", and
"a nice trick".

Describe rather than evaluate. Say what a choice does and what it trades away, but
do not rule on whether it was right. "The right choice", "the correct way", "the
proper unit", "this is the fix", "you should", and "the best approach" all need
recasting into what the choice does and what the alternative would have done.

Do not restate the code line by line. "This groups by lemma and sorts descending"
is already visible in `group_by("lemma").sort(...)`. Use the library's own
vocabulary rather than inventing a synonym for it: the parameter is `file_id_column`,
so the things it names are files, not documents or parts.

## Formatting

Unlike a reference page, a notebook may use the full markdown repertoire, and the
existing notebooks do:

- `##` and `###` headings at the seams of the arc. These build the page's table of
  contents in the rendered site, so a notebook of any length needs them.
- Bullet lists for enumerating measures, columns, or the entries of a `method`
  argument. Prose stays the backbone; a list is for things that genuinely are a list.
- Tables for a corpus's columns and what they hold, as in `frequencies.ipynb`.
- Bold on a term at first use, italics for cited word forms (*the*, *hapax legomena*).
- Math via `$$...$$` and `$...$`, rendered by katex. Both delimiters are configured
  in `docs/javascripts/katex.js`. A formula in a notebook is there to be read
  alongside the call that computes it, so give the symbols the column names the
  result will use.
- Author-date citations, with a **References** list in the title cell or a bulleted
  one under the section that uses them.

No emoji.

Links to a reference page must be written as **built URLs, not source paths**.
`mkdocs` rewrites `.md` links only on markdown pages; `mkdocs-jupyter` passes a
notebook's markdown through untouched, and `--strict` does not catch the
difference. A notebook renders to `/notebooks/<name>/` and a reference page to
`/<name>/`, so the link from a notebook is two levels up and has no extension:

    [Keywords](../../keywords/)          correct
    [Keywords](../keywords.md)           builds clean, 404s on the site

The same holds for a link to a sibling notebook: `[Concordances](../concordance/)`.
Build and follow any link you add -- this one silently produces a dead link.

If the notebook has no title cell, add one: an `#` heading, a paragraph on what the
analysis is and what the notebook works through, and a numbered list of the moves.
`frequencies.ipynb` cell 0 is the model.

## Numbers and claims

The notebook cannot be re-run -- `mkdocs.yml` sets `execute: false`, and the corpora
under `data/` are gitignored and far too large to commit. Every number in the prose
therefore has to come from the committed output of a cell, and every claim about
behavior from reading `python/polars_corpus/`. A note that is merely plausible is
worse than none.

Those corpora are usually sitting on the machine all the same, so a claim about
behavior can be checked instead of reasoned about: read the parquet in a scratch
script. That is how to settle what a tag regex really matches, or to
reproduce a number a cell printed -- recomputing a measure from the corpus and
landing on the committed value confirms the mechanism and the reading of it at once.
The corpus is for checking claims, never for sourcing numbers: what goes in the prose
still has to be visible on the page, since that is all the reader can check. And
checking is not running -- the notebook itself stays untouched.

Quote figures the way `frequencies.ipynb` does -- "3,850,429 tokens and 143,502
types" -- copied from the output above, not recomputed or rounded from memory.

Read the plots, not only the tables. The committed `image/png` outputs decode to real
images that can be opened and looked at, and any note that reads a figure needs that
-- where the curves cross, what the axis ranges are, which point sits on its own.
Extracting them to the scratchpad first also leaves a baseline for the output check
below.

Watch for output that has gone stale: a cell whose source was edited after it last
ran, or whose output records an older signature. `concordance.ipynb` has one, an
error output naming a keyword argument the current `search()` does not take. Do not
write prose that explains a stale output as though it were current behavior, and do
not delete or fix it. Leave it and name it in the report.

Watch too for a figure whose label contradicts the code that drew it -- a `label=` or
a title naming one text while the cell computes another. Nothing in the image gives
that away, so it is caught only by reading the call. Write the note from what the
code does and name the mismatch in the report.

Worth checking rather than assuming, since these are what the notes get wrong:

- whether a call returns eager or lazy, and whether the notebook collected it;
- row order, wherever a method can reorder;
- which rows a measure drops -- nulls, rows under `min_freq`, files with no hits;
- what a column in the output is actually named, character for character;
- what the Rust in `src/` decides, for anything about matching or context windows.

## Mechanics

Edit with `nbformat`, and insert only new markdown cells. Never modify, reorder, or
delete a code cell, never change `execution_count`, never strip or regenerate
outputs, and never run the notebook.

Edit in place. The notebooks are tracked, and `mkdocs.yml` names each file in the
nav, so a `-annotated` copy would have to be renamed back anyway. Run `git diff`
on the notebook first: if it already has uncommitted changes, the annotation will
be mixed in with them, which is fine but belongs in the report -- note the diff's
deletion count while you are there. A notebook that is new to `docs/notebooks/` also
needs a line added to the **Examples** section of the nav in `mkdocs.yml`.

Because the edit is in place, take the baseline before the first insertion: copy the
notebook to the scratchpad. `git show HEAD:<path>` is not a substitute. A notebook
re-run since its last commit differs from HEAD in every execution count and most
outputs, so that comparison reports differences the pass did not make and hides the
ones it did.

Verify before reporting:

```python
import nbformat

orig = nbformat.read(baseline, as_version=4)  # the copy taken before editing
new = nbformat.read(path, as_version=4)       # the notebook, edited in place
o = [c for c in orig.cells if c.cell_type == "code"]
n = [c for c in new.cells if c.cell_type == "code"]
assert len(o) == len(n), f"code cell count changed: {len(o)} -> {len(n)}"
for a, b in zip(o, n):
    assert a.source == b.source, "code cell source modified"
    assert a.get("outputs") == b.get("outputs"), "outputs modified"
    assert a.get("execution_count") == b.get("execution_count"), "execution_count changed"
nbformat.validate(new)
```

With no baseline, `git diff --stat` still catches the failure that matters: inserting
markdown adds lines and removes none, so the deletion count has to be exactly what it
was before the pass.

Then `mkdocs build --strict` and confirm the notebook adds no warnings, that the
headings appear in the page's table of contents, that any math rendered, and that
every link you added resolves to a file that exists under `site/`. The build is
warning-free today apart from a run of "Div ... unclosed" notices out of
`mkdocs-jupyter`'s own conversion, which predate any annotation and are not
something to chase.

If the author re-runs the notebook after the pass, check every quoted number against
the new outputs before treating the work as done. Figure bytes change on a re-run
even when nothing in the content does, because point plotting order follows
`group_by`, so compare what the note claims rather than the checksum.

## Deciding rather than asking

Do not open with a round of questions. Where the intent of a stretch of cells is
unclear, settle it from the source, the sibling notebooks, and the reference pages,
then write the note under the reading you settled on and flag it in the report. A
note that is wrong in a named way is easy to fix; a pass that stalled waiting for an
answer is not.

The exception is a notebook that breaks off mid-analysis, where the payoff is
missing rather than merely unstated. Annotate everything up to that point, leave the
remainder alone, and ask about that one thing at the end.

## Reporting back

Close with a short account: how many cells got notes, which two or three are
guesses and what they assume, any stale or dead cells found, and anything left
mid-thought that needs a sentence from the author. Do not summarize the notebook's
content back to them -- they wrote it.
