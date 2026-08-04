# Simple Query Language Implementation Status

**Last updated:** 2026-08-04
**Status:** Feature-complete except proximity operators

The Simple query language is the BNCweb-style syntax that `search()` accepts by
default. `simple_parser.simple_to_cqp()` translates it directly to a CQP string,
which is then compiled by the same matcher that backs `search_cqp()` — there is
no intermediate AST and no separate execution path.

The grammar is defined in **lark** (`_GRAMMAR` in `simple_parser.py`) and
compiled once at import as an LALR parser; each call instantiates a
`SimpleCompiler` transformer with the requested column names, whose methods emit
CQP fragments. An earlier implementation used pyparsing and rebuilt its grammar
per call; it was replaced, so any description of `Forward()`, `Combine`, or
parse actions no longer applies. `simple_grammar.md` holds the grammar
specification and `Simple_query_language.pdf` the original BNCweb documentation.

Covered by 79 tests in `python/tests/test_simple_query.py`.

---

## Supported Syntax

| Feature | Syntax | Example | Compiles to |
|---|---|---|---|
| Basic word | `word` | `fox` | `[token="fox"%c]` |
| Wildcards | `?` `*` `+` | `s?ng`, `*able` | `[token="s.ng"%c]` |
| Alternatives | `[a,b]` | `[car,truck]` | `[token="car\|truck"%c]` |
| Sequences | `a b` | `quick brown fox` | three tokens |
| Gaps | `*` `+` | `fox * over` | `[]?` between tokens |
| Repeated gaps | `++` `***` | `++` | `[]{2}` |
| POS tags | `word_TAG`, `_TAG` | `lights_NN2`, `_PNX` | `& pos="NN2"` |
| Simplified POS | `_{CLASS}` | `{box}_{SUBST}` | `pos="N.*"` |
| Lemmas | `{lemma}` | `{light}` | `[lemma="light"%c]` |
| Lemma + class | `{lemma/POS}` | `{light/V}` | `& pos="V.*"` |
| Lemma + tag | `{lemma}_TAG` | `{walk}_VBD` | `& pos="VBD"` |
| Groups | `(...)` with `? + * {m,n}` | `(very)? big`, `(quick){2}` | `(...)?`, `(...){2}` |
| Disjunction | `(a \| b \| c)` | `(a \| b \| c)` | `([..]\|[..]\|[..])` |
| Variable bindings | `$name: pattern` | `$x: fox` | `$x: ([token="fox"%c])` |

Searches are case-insensitive by default (`%c`). Column names are configurable
throughout: `token_column`, `pos_column`, `lemma_column`.

### Simplified POS classes

| Class | Pattern | Matches |
|---|---|---|
| `V`, `VERB` | `V.*` | VB, VBD, VBZ, VVI, ... |
| `N`, `SUBST` | `N.*` | NN, NNS, NN1, NN2, ... |
| `A`, `ADJ` | `(AJ.*\|JJ.*)` | AJ0, JJ, JJR, JJS |
| `ADV` | `(AV.*\|RB.*)` | AV0, RB, RBR |

Chosen to cover both BNC CLAWS-5 and Penn Treebank tagsets.

---

## Not Implemented

### Proximity operators

`<<s>>`, `<<3>>`, `<<5<<`, `>>5>>` are not in the grammar; they raise
`UnexpectedCharacters`. This is the last substantial feature.

There is no direct CQP equivalent, so it needs one of:

1. **Expand to CQP disjunctions.** `day <<3>> night` becomes
   `([token="day"%c] []{0,3} [token="night"%c]) | ([token="night"%c] []{0,3} [token="day"%c])`.
   Simple, but combinatorial once constraints nest, e.g.
   `waste <<s>> (time <<3>> money)`.
2. **Add proximity opcodes to the Rust matcher.** Efficient and the right
   long-term answer, but the largest change.
3. **Filter after matching.** Search each term separately and post-process by
   distance. Least efficient, and awkward to compose.

Sentence-level proximity (`<<s>>`) additionally needs a sentence boundary
column, which `with_chunk_index()` can already supply.

### Embedded alternatives

`neighbo[u,]r` is meant to match *neighbour* and *neighbor*. It currently
**parses without error and produces the wrong query** — three separate tokens,
`[token="neighbo"%c] [token="u|"%c] [token="r"%c]` — rather than failing
loudly. Worth rejecting explicitly until it is supported.

Workaround: `[neighbour,neighbor]`.

---

## Known Wrinkles

- Escaping (`\?`, `\*`) is supported but has no dedicated tests.
- One comment in `simple_parser.py` still refers to the pyparsing
  implementation.
