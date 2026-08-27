# Feature Gaps - polars-corpus

**Last updated:** 2026-08-26
**Scope:** what to add to cover the basic corpus-analysis toolkit

This is a survey of the documentation -- the twelve reference pages, the five
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

## 1. Collocation

Mostly closed since the last survey. `collocations()` takes a search and
returns its collocates ranked by one or more association measures, which is
the path from a node word to a ranked list that nothing showed before;
`docs/collocations.md` documents it and is on the nav. log-dice and MI3
landed alongside t-score and z-score, so the two most-reported scores in the
literature are no longer missing. Windows take an asymmetric `(left, right)`
pair for L5/R5 spans, and a `chunk_column` runs the window to the edges of
the chunk holding the match instead, so a sentence tag column gives a span
that stops at the sentence boundary.

What is left:

- **A notebook.** `docs/notebooks/collocation.ipynb` is a working draft --
  unannotated, not in the nav, and not counted as documentation. Until it is
  written the reference page carries the whole topic, and every other measure
  in the library has a notebook making the case for when to reach for it.
- **Colligation.** Collocates over the `pos` column rather than `token`, or
  over a struct of both. `expr=` already reaches it -- the draft notebook does
  exactly this -- but no reader will discover it from the reference page
  alone. It is a worked example, not a feature.
- **Collocation networks.** Still a TODO in `visualizations.py`.

Dependency-based collocation (word sketches) is the one major collocation
feature nothing in the docs reaches. It depends on the corpus carrying
dependency columns, which is a data-format decision -- see section 7.

## 2. Frequency lists as a function

`frequencies.ipynb` is good teaching, but every cell hand-rolls the same
pipeline: `filter(is_letters) -> to_lowercase -> group_by -> agg(pl.len()) ->
sort`. There is no `frequency_list()`.

One call returning type, count, relative rate and document range, taking
`min_freq` and a `by=` grouping column for subcorpus or diachronic
breakdowns, would be the most-used function in the library and would retire
the six-line incantation students currently copy.

Related and also absent from the docs: **stopword lists**, and any
normalization helper beyond `is_letters` -- case folding is manual everywhere
it appears.

## 3. Concordance workflow past generating one

`SearchResults.concordance()` returns list columns; `concordance.ipynb` then
reaches for `pl.col('token').list.first()` to sort by L1 and a manual
`group_by('text_type').len()` to break hits down by metadata. Standard in
every concordancer and missing here:

- **Sort by context position** (L1/L2/R1...), the classic KWIC sort.
- **Thinning and random sampling** of hits, reproducibly.
- **Breakdown of hits across a metadata column.**
- **Export** to CSV or HTML. Nothing in the docs mentions saving anything.
- The interactive `ConcordanceWidget` is documented only in
  `DEVELOPMENT_STATUS.md`, not on the docs site.

## 4. N-grams and clusters

`ngrams` sits in `docs/utils.md` with no prose and no example. There is no
clusters-around-a-node tool and no lexical-bundle extraction, which is the
staple of the phraseology side of the field.

## 5. Keyness effect sizes

`keywords()` offers `ll` and `ttest`. The effect-size measures the field now
expects alongside significance are missing: **log ratio** (Hardie 2014),
**%DIFF**, odds ratio, and Bayes-factor BIC. The keywords notebook already
makes the argument for why $G^2$ alone misleads, so the ground is prepared.
`logdice` arrived with the collocation measures and is an effect size, but it
is not offered as a `keywords()` method.

## 6. Text-level descriptive measures

Everything documented is token-level or type-level. Nothing computes per-text
**sentence length, mean word length, or readability** (Flesch, ARI) -- the
basic descriptive battery. `chunk_id()` and `with_chunk_index()` supply
sentence boundaries but appear nowhere on the docs site, which also blocks
anything sentence-scoped.

## 7. `from_spacy()`

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
   Either pick a conservative default set (`token`, `lemma`, `pos`, `tag`,
   `sentence_tag`, `file_id`) with the rest opt-in, or treat `from_spacy()` as
   the moment we decide whether the library wants dependency columns at all.

`from_stanza()` is the obvious sibling. spaCy first; it has the users.

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

1. A real `frequency_list()` (section 2).
2. Concordance sorting, sampling and export (section 3).
3. The collocation notebook (section 1), the last piece of what was the
   largest hole.

These are what a student opens AntConc for, and the docs show only collocation
so far. `from_spacy()` (section 7) is the fourth, and the one that most widens
who can use the library at all.

Note how much of items 2 and 3 is writing rather than coding. That is not a
discount on the work -- it is where the work is. Collocation is the evidence:
the function was a few hundred lines, and it still is not shipped until the
notebook exists.
