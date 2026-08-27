# Feature Gaps - polars-corpus

**Last updated:** 2026-08-27
**Scope:** what to add to cover the basic corpus-analysis toolkit

This is a survey of the documentation -- the twelve reference pages, the six
example notebooks, `mkdocs.yml`, the README -- against what a linguist expects
a corpus tool to do.

A feature that isn't documented is a feature users don't have. Several entries
below are already implemented and merely unwritten; they are listed as gaps
anyway, on equal footing with the ones that need code, because from outside
the repo there is no difference. Writing the page is the remaining work, and
until it exists the feature isn't shipped.

`DEVELOPMENT_STATUS.md` covers what works and what is broken. This covers what
isn't there yet.

---

## 1. Collocation -- closed

`collocations()` takes a search and returns its collocates ranked by one or
more association measures; `docs/collocations.md` documents it,
`docs/notebooks/collocation.ipynb` is annotated and on the nav, and log-dice
and MI3 landed alongside t-score and z-score, so the two most-reported scores
in the literature are no longer missing. Windows take an asymmetric
`(left, right)` pair for L5/R5 spans, and a `chunk_column` runs the window to
the edges of the chunk holding the match instead, so a sentence tag column
gives a span that stops at the sentence boundary. A measure the library does
not ship goes in the same argument as one it does, as a callable.

Three things are deliberately out rather than pending:

- **Colligation** stays a worked example, not a top-level function. `expr=`
  already reaches it -- collocate over `pos`, or over a struct of token and
  tag -- and the notebook does exactly that. A `colligations()` would be a
  rename of an argument.
- **Visualizing collocations** is a task of its own, not the tail of this one.
  Collocation networks in particular are not the direction; clustering the
  collocate space, probably spectral, is the shape worth trying. Until then
  the TODO in `visualizations.py` is a note for later, not a documentation
  gap.
- **Dependency-based collocation** (word sketches) needs dependency-annotated
  corpora, and that annotation is a separate project rather than something
  this library will grow. It may be linked from here later. See section 7.

## 2. Frequency lists -- closed

`frequency_list()` takes a corpus and returns one row per type, with the count
in `freq`, the count as a rate per `basis` words in `rate`, and the number of
files the type occurs in in `range`. The rate divides by the tokens actually
counted, so filtering the result afterwards drops rows without moving the rate
on the rows that remain.

`docs/frequencies.md` documents it and is on the nav, and
`docs/notebooks/frequencies.ipynb` now leads with the function, keeping the
hand-rolled `group_by` once as the explanation of what it packages.

Three things are deliberately out rather than pending:

- **A `by=` argument** for subcorpus or diachronic breakdowns. Getting it from
  Polars' own `group_by` instead needs `frequency_list` to return an
  expression, which was tried: it works, and the `.over()` aggregations are
  correctly scoped inside `agg`, but it cannot sort itself, costs the caller
  `.explode().unnest()` at every call site, runs about twice as slow at 20M
  rows, and has no frame to check column names against. The frame form matches
  `dispersion` and `keywords`; the breakdown can come back as `by=` later
  without changing anything already shipped.
- **Arguments that duplicate a frame operation.** `lowercase=`,
  `letters_only=`, `min_freq=` and `min_range=` were all written, and all
  removed once it was clear each had an exact equivalent the caller could
  already write. Normalizing is `expr`'s job -- case folding is
  `pl.col("token").str.to_lowercase()`, and a token `expr` evaluates to null is
  dropped, so restricting what counts as a word is a `filter` on the corpus or
  a `when`/`then`. Thresholding is `.filter(pl.col("freq") >= 10)` on the
  result, which is the same rows in the same order, because the rate is
  computed before either. The earlier survey asked for `min_freq` and for a
  normalization helper by name; the answer to both is that the frame and the
  expression were already the helper.

  `dispersion` and `collocations` keep their `min_freq`, which is not the same
  argument: there it guards a measure that misbehaves on rare words, rather
  than tidying a result. A count of 1 is a perfectly good count.
- **Stopword lists**, which the earlier survey listed here as missing. They are
  an information-retrieval device rather than a linguistic one: the list is a
  judgement about which words carry no meaning, made once, for no particular
  question, and then applied to every question. A student who drops *not*, *no*
  and *very* because a list says so has been taught to throw away the data
  before looking at it. Whoever wants one can write `is_in` over their own
  list, which is also the version they can defend.

## 3. Concordance workflow -- one page still owed

`kwic()` sorts a concordance. It takes a position -- `"L1"`, `"L2"` and so on
to the left, `"R1"`, `"R2"` to the right, `"node"` for the match itself -- and
returns the expression that selects the token sitting there, so
`conc.sort(plc.kwic("L1"), plc.kwic("L2"))` is the classic KWIC sort. The
signed integers CQP writes those positions as work too: `-1` and `1` either
side of `0`. Being an expression rather than a sort argument, it groups and
filters on the same footing -- `conc.group_by(plc.kwic("R1")).len()` is what
follows the node -- and it composes, so case folding is
`plc.kwic("L1").str.to_lowercase()` and stays the caller's, exactly as
normalizing is `expr`'s job in `frequency_list`. A position the context does
not reach comes out null, and `nulls_last=True` sends those lines to the end
instead of the top.

`concordance(as_str=True)` exports. The blocker there was concrete rather than
architectural: the `List(String)` columns are exactly what makes a concordance
computable, and exactly what CSV refuses -- `write_csv` on a concordance fails
outright with `ComputeError: CSV format does not support nested data`.
`as_str=True` joins each of them -- the match, both contexts, every `$var:`
binding column -- into one space-separated string, leaving `metadata` scalars
and any `List(Struct)` column alone. The name is the one `ngrams()` already
uses for the same switch. Joining is the last step rather than the default,
because the lists are what `kwic` and `ConcordanceWidget` read; sorting and
exporting in one breath therefore ends in the line the argument cannot supply:

```python
conc.sort(plc.kwic("L1")).with_columns(cs.by_dtype(pl.List(pl.String)).list.join(" "))
```

`docs/search.md` documents both, and `docs/notebooks/concordance.ipynb` is
annotated and on the nav: it counts the words either side of a node, sorts on
them, writes the lines out as CSV and as a `great_tables` table, and finishes
on the breakdown below.

**Thinning and random sampling** was listed here in error. `sample(k, seed=)`,
`shuffle(seed=)`, `head` and `tail` were already on `_SearchResultsBase` when
the survey was written. All four return `Self`, so they chain into the rest of
the workflow, and `sample` and `shuffle` put the global random state back
afterwards, so seeding a sample for reproducibility does not quietly reseed
everything else in the notebook. The entry was simply wrong.

Two things are deliberately out rather than pending:

- **A breakdown of hits across a metadata column.** The raw counts are
  `conc.group_by("text_type").len()`, which is Polars' and needs no wrapping.
  The version worth having is normalized against how big each category is:
  raw hit counts across categories of unequal size mislead in exactly the way
  the keywords notebook argues $G^2$ misleads, and the raw table also drops
  any category the query found nothing in, since a category with no hits
  contributes no concordance rows -- *shall* in the notebook loses `voyage`,
  the largest genre in the corpus. The primitive for the better answer is
  already here -- `with_spans_as_chunks()` writes the matches back onto the
  corpus as BIO tags, so a single `group_by` over the tagged corpus counts
  hits and tokens together and the rate falls out of the same aggregation.
  That is a notebook cell, not a function.
- **HTML export.** `great_tables` is already in the `examples` extra and
  already how the example notebooks render a KWIC table; after `as_str=True`
  the whole recipe is `conc.style`. A function wrapping a `.style` call would
  buy a dependency question and nothing else.

What is left is a page. The interactive `ConcordanceWidget` is documented only
in `DEVELOPMENT_STATUS.md` and appears nowhere on the docs site, which by the
standard at the top of this file makes it a feature users do not have. It is
the only thing this section still owes.

## 4. N-grams and clusters

`ngrams` sits in `docs/utils.md` with no prose and no example. There is no
clusters-around-a-node tool and no lexical-bundle extraction, which is the
staple of the phraseology side of the field.

## 5. Keyness effect sizes

`keywords()` offers `chisq`, `ll`, `mi3`, `minsens`, `pmi`, `smp`, `tscore`,
`ttest`, and `zscore` -- `mi3`, `tscore`, and `zscore` joined the list when the
collocation measures landed. The effect-size measures the field now expects
alongside significance are still missing: **log ratio** (Hardie 2014),
**%DIFF**, odds ratio, and Bayes-factor BIC. The keywords notebook already
makes the argument for why $G^2$ alone misleads, so the ground is prepared.

`logdice` is deliberately not among them. In the keyness table the second
marginal `f2` is the size of the target corpus, not a second word's frequency,
so `2 f12 / (f1 + f2)` is dominated by `f2` and log-Dice reduces to a
monotone function of the word's relative frequency in the target. It is an
effect size for collocation, where both marginals are word frequencies; here
it would just rank by frequency.

A reader who knows the formula can now pass it in: `method=` takes a callable
over `(f12, f1, f2, n)`, and log ratio is four lines. That is a workaround,
not a fix -- the gap is what a student finds in the list of methods, and
these belong in it.

## 6. Text-level descriptive measures

Everything documented is token-level or type-level. Nothing computes per-text
**sentence length, mean word length, or readability** (Flesch, ARI) -- the
basic descriptive battery. `chunk_id()` and `with_chunk_index()` supply
sentence boundaries but appear nowhere on the docs site, which also blocks
anything sentence-scoped.

## 7. `from_spacy()` -- after the next release

The documented I/O story is `read_text_corpus()` / `scan_text_corpus()` for
plain text and `from_nltk()` for corpora somebody else already annotated.
Nothing covers the path most users actually take: raw text they have to
annotate themselves. spaCy is the default answer, so `from_spacy()` closes the
gap between "I have a directory of text files" and "I have a DataFrame with
`token`, `pos`, `lemma`". It is worth more than `from_nltk()` in practice.

It also fills in columns the rest of the library already consumes but has no
documented way to produce:

- **`lemma`.** The Simple query language has lemma syntax (`{walk}_VB*`, a
  whole section of `docs/simple_query.md`) and `dispersion.ipynb` queries a
  `lemma` column, but no documented reader creates one. Today it arrives only
  if your Parquet already had it.
- **Sentence boundaries.** `sentence_tag` / `with_chunk_index()`, which the
  roadmap says sentence-level proximity (`<<s>>`) will need. spaCy segments
  sentences for free.
- **A simplified POS class beside the fine tag.** The Simple language's
  `_{SUBST}` classes against `_NN1` tags; spaCy hands over both `pos_` (UPOS)
  and `tag_`.

Two decisions shape the API more than the conversion does:

1. **Batch, not one `Doc`.** The natural argument is an iterable of `Doc`s --
   what `nlp.pipe()` returns -- with file IDs alongside, whether as a parallel
   iterable, `(file_id, Doc)` pairs, or `Doc.user_data`. A `from_spacy(doc)`
   taking a single document pushes users into a per-file loop and a
   `pl.concat`, which is the boilerplate this is meant to remove.
2. **Which columns, under what names.** spaCy offers far more than the data
   format in `CLAUDE.md` describes: `head` and `dep_`, `ent_type_`, `morph`,
   `is_stop`, whitespace. Our format is flat -- `token`, `pos`/`c5`, `mode`,
   `file_id` -- with no notion of dependency arcs and no tool that reads them.
   That question is settled for the arcs: dependency annotation belongs to a
   separate project, not to this one, so `head`/`dep_` stay out. What is left
   is picking a conservative default set (`token`, `lemma`, `pos`, `tag`,
   `sentence_tag`, `file_id`) and deciding which of the rest are opt-in.

`from_stanza()` is the obvious sibling. spaCy first; it has the users.

This is deferred past the next release, for the reason the widget's page is:
it is a project of its own -- a dependency, a column-naming decision, a batch
API and a body of tests that need spaCy installed to run -- and nothing
already shipped is waiting on it. It widens who can use the library more than
anything else in this file, which is an argument for doing it properly rather
than for doing it next.

## 8. The query language and the missing guide

One genuine code gap and two pages nobody has written. By the standard
above, they rank the same.

- **Proximity operators** (`<<3>>`, `<<s>>`) -- the biggest query-language
  gap, already on the roadmap. Sentence-bounded collocation windows now come
  from `collocations(chunk_column=...)` instead, so this no longer pays
  twice; it is a query-language gap and nothing else.
- **No narrative getting-started guide.** The nav entries are commented out in
  `mkdocs.yml`.
- **`assoc.md`, `lexical.md` and `utils.md` are bare mkdocstrings stubs** --
  no prose on what a measure means or when to reach for it, unlike
  `dispersion.md`, which at least links its notebook.

---

## If only three

1. ~~A real `frequency_list()` (section 2).~~ Done.
2. ~~Concordance sorting, sampling and export (section 3).~~ Done.
3. Keyness effect sizes (section 5): log ratio and %DIFF, the two the field
   actually reports.

The first two are what a student opens AntConc for, and both are covered now:
frequency, collocation and the concordance workflow each have their function,
their reference page and their notebook. The third is the smallest piece of
code left with the largest claim on being taken seriously. `keywords()` ranks
by significance alone, which the keywords notebook itself spends a section
arguing is not enough, and `_apply_measure` already takes a callable over
`(f12, f1, f2, n)` -- so each measure is four lines, a name in the list, and a
paragraph saying when to reach for it.

Two things sit below that line rather than above it, and are deliberately
after the next release:

- **`from_spacy()` (section 7).** It was item 3 here, and it is still the
  entry that most widens who can use the library at all. But it is a project
  with its own dependency, its own API decisions and its own tests, and
  nothing already shipped is waiting on it.
- **A page for `ConcordanceWidget` (section 3).** The same shape at a smaller
  scale: the widget is written and tested, and what it needs is a page and a
  place in a notebook -- a task with its own beginning and end rather than the
  tail of this one.

Note how much of item 2 turned out to be writing rather than coding: sampling
was implemented before it was ever listed as missing, and what section 3 still
owes is a page, not a function. That is not a discount on the work -- it is
where the work is. Collocation is the evidence: the function was a few hundred
lines, and it was not shipped until the notebook existed.
