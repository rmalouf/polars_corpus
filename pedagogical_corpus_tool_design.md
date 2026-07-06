# Notebook Corpus Toolkit: Design Notes

## Purpose

Build a notebook-based corpus analysis toolkit for teaching. The toolkit should use Polars directly where Polars is already strong, and add corpus-specific concepts where Polars has no opinion.

Target environment:

- Jupyter notebooks
- instructional cluster
- small local GPUs
- student access to ChatGPT/Gemini by copy-paste, not API

Core principle:

> Polars is the computation layer. The corpus toolkit supplies schema roles, corpus transforms, inspectable recipes, and teaching-oriented result objects.

---

## Non-Goals

Do not build:

- a replacement dataframe API
- a second predicate/filtering DSL
- black-box corpus methods that only return tables
- hidden AI calls
- elaborate wrappers for ordinary Polars operations

Filtering, comparison, membership, date ranges, grouping, joining, sorting, string operations, and lazy execution stay in Polars.

---

## API Layers

### Layer 0: Polars

Users should write ordinary Polars expressions.

```python
c = corpus.cols

freq = (
    corpus.tokens()
    .filter(c.pos.is_in(["NOUN", "PROPN"]))
    .filter(c.date.is_between(1900, 1950))
    .group_by(c.lemma)
    .len()
)
```

### Layer 1: Corpus Schema Roles

Corpus metadata should live in a sidecar schema owned by the corpus wrapper. The schema maps semantic roles to physical column names.

```python
corpus = Corpus(
    tokens=tokens_df,
    documents=docs_df,
    schema={
        "tokens": {
            "token": "word",
            "lemma": "lemma",
            "pos": "upos",
            "doc_id": "doc_id",
            "date": "year",
        },
        "documents": {
            "doc_id": "doc_id",
            "genre": "register",
            "text": "text",
        },
    },
)
```

Accessors return ordinary Polars expressions:

```python
c = corpus.cols

corpus.tokens().filter(c.token.str.len_chars() > 3)
corpus.tokens().filter(c.lemma.is_in(["risk", "danger"]))
corpus.tokens().filter(c.pos != "PUNCT")
```

Rule:

- If a helper merely abbreviates a Polars operator, do not add it.
- If a helper names a corpus role or linguistic convention, consider it.

### Layer 2: Corpus Transforms

Composable operations over Polars data.

```python
count_items(corpus.tokens(), item=c.lemma, by=c.genre)
context_windows(corpus.tokens(), node="risk", span=(-5, 5))
contingency(feature=c.lemma, context=c.genre)
association(table, measure="log_likelihood")
```

These should make explicit:

- item counted
- unit of analysis
- comparison context
- raw counts
- expected counts
- association measure

### Layer 3: Named Recipes

Convenience workflows for standard corpus analyses.

```python
kw = corpus.keywords(
    target=c.genre == "student_essay",
    reference=c.genre == "published_article",
    item=c.lemma,
    measure="log_ratio",
)

coll = corpus.collocations(
    node="risk",
    item=c.lemma,
    span=(-5, 5),
    measure="log_dice",
)
```

Recipes should be readable, but not opaque. They should expose their lower-level steps.

### Layer 4: Result Objects

High-level methods return result objects, not bare dataframes.

```python
kw.show(20)
kw.df
kw.steps()
kw.intermediate("contingency_table")
kw.explain_row("however")
kw.examples("however")
kw.audit()
kw.as_polars()
kw.to_markdown()
kw.to_ai_prompt(...)
```

The dataframe is available, but the object also carries parameters, provenance, intermediate tables, examples, and audit information.

---

## Shared Association Framework

Keywords, collocations, and collexemes should share an underlying association model.

```text
feature + context
-> observed counts
-> expected counts
-> association measure
-> ranked results
-> examples
-> audit checks
```

Examples:

```text
keywords:
    feature = lemma
    context = target corpus vs reference corpus

collocations:
    feature = lemma
    context = near node vs elsewhere

collexemes:
    feature = filler
    context = construction slot vs other contexts
```

Use a common `AssociationResult` where possible.

```python
kw = corpus.keywords(...)
coll = corpus.collocations(...)

assert type(kw) == AssociationResult
assert type(coll) == AssociationResult
```

---

## Recipe Expansion

Recipes should be expandable into conceptual pseudocode.

```python
kw.expand()
```

Example output:

```python
c = corpus.cols

target_freq = target.group_by(c.lemma).len()
reference_freq = reference.group_by(c.lemma).len()

table = make_2x2_table(
    target_freq,
    reference_freq,
    item=c.lemma,
)

result = association(table, measure="log_likelihood")
```

The expansion need not reproduce internal optimizations. It should show the method structure.

---

## Row-Level Explanation

Row-level explanation is central for teaching.

```python
kw.explain_row("however")
```

Should show:

- raw counts
- normalized frequencies
- 2x2 table
- expected counts
- score calculation
- examples
- cautions

For collocations, explanation should include node, collocate, span, near-node counts, elsewhere counts, and example lines.

---

## Auditing

Each major result should support `audit()`.

Useful checks:

- corpus size imbalance
- date, genre, source, author, assignment, or topic confounds
- document concentration / dispersion
- measure sensitivity
- low-frequency dominance
- span sensitivity for collocations
- function word inclusion
- zero-count handling
- raw vs normalized frequency visibility

The goal is not to prevent mistakes. The goal is to make them visible.

---

## Assignment Modes

### Construction Assignments

Students rebuild methods with Polars plus corpus transforms.

```python
target_freq = target.group_by(c.lemma).len()
reference_freq = reference.group_by(c.lemma).len()
table = make_contingency_table(target_freq, reference_freq)
scores = association(table, measure="log_likelihood")
```

Purpose: teach counts, denominators, expected values, and association measures.

### Application Assignments

Students use named recipes and interpret results critically.

```python
kw = corpus.keywords(target, reference, item=c.lemma)
coll = corpus.collocations("risk", span=(-5, 5))

kw.show(20)
kw.explain_row("however")
kw.audit()
coll.examples("significant")
```

Purpose: teach interpretation, evidence selection, robustness checks, and limitations.

---

## Teaching Pattern

For each major method:

```text
1. Use the recipe.
2. Inspect the steps.
3. Rebuild a simplified version.
4. Use the recipe critically.
```

Example:

```python
kw = corpus.keywords(target, reference, item=c.lemma)
kw.show(20)
kw.steps()
kw.intermediate("contingency_table")
kw.explain_row("however")
kw.audit()
```

---

## AI Workflows

Students have access to AI tools by copy-paste. The toolkit should support explicit prompt packets and validation, not hidden API calls.

Workflow:

```text
corpus evidence
-> prompt packet
-> copy into ChatGPT/Gemini
-> paste response into notebook
-> validate response
-> compare against evidence or student labels
```

Prompt packets should include:

- task description
- allowed labels
- row ids
- text hashes
- required output format
- evidence restrictions
- instructions not to invent context

Validation should catch:

- missing ids
- extra ids
- duplicate rows
- invalid labels
- malformed JSON
- changed text hashes
- hallucinated examples

Example:

```python
prompt = kwic.to_ai_prompt(
    task="classify_discourse_function",
    labels={
        "CONTRAST": "marks contrast",
        "CONCESSION": "acknowledges then redirects",
        "UNCLEAR": "insufficient context",
    },
    output_format="jsonl",
)

labels = corpus.read_ai_labels("labels.jsonl")
labels.validate_against(prompt)
labels.agreement_with(student_labels)
labels.inspect_disagreements()
```

---

## Local GPU Features

First priority: local embeddings for semantic concordance work.

```python
risk = corpus.kwic("risk", window=30)

clusters = (
    risk
    .embed_contexts()
    .cluster(n_clusters=6)
    .label_with_keywords()
)

clusters.summary()
clusters.inspect(cluster=2)
clusters.to_ai_prompt(task="name_clusters")
```

Pedagogical point: embeddings help find and group relevant contexts, but clusters still require human interpretation.

---

## Claim Cards

Final-project claims can be represented as evidence-linked objects.

```python
claim = corpus.claim(
    "Student essays frame climate policy more as personal responsibility than published articles do."
)

claim.add_evidence(keywords)
claim.add_evidence(collocates)
claim.add_counterevidence(counter_kwic)
claim.add_ai_critique(ai_response)
claim.render()
```

A claim card should contain:

- claim
- evidence ids
- counterevidence
- audit status
- limitations
- current confidence/status

This pushes students from table production toward evidence-based argument.

---

## Build Priorities

1. Sidecar schema and column-role accessors
2. Core corpus views: `tokens()`, `documents()`, `sentences()` if needed
3. Corpus transforms: counts, context windows, contingency tables, association measures
4. Result objects: `show`, `steps`, `intermediate`, `explain_row`, `examples`, `audit`
5. Recipes: keywords, collocations, n-grams / lexical bundles, dispersion
6. AI prompt packets and response validation
7. Annotation comparison tools
8. Semantic KWIC with local embeddings
9. Claim cards

---

## Open Design Questions

- Should column accessors be global (`pc.lemma`) or corpus-specific (`corpus.cols.lemma`)? Current preference: corpus-specific.
- Should recipes accept only Polars expressions, or also named subcorpora? Current preference: accept both, but keep expressions primary.
- How much should `expand()` show: executable code or conceptual pseudocode? Current preference: conceptual pseudocode.
- How strict should schema validation be? Current preference: validate required roles and warn on questionable dtypes.
- Which association measures should ship first? Current preference: log likelihood, log ratio, logDice, chi-square.
