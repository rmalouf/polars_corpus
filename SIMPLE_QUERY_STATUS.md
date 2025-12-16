# Simple Query Language Implementation Status

## Current Status: Phase 3 Partially Complete ✅

Last updated: 2025-01-28

## What's Done (Phase 1)

### Implementation
- ✅ **`simple_parser.py`** - Clean, functional parser with no AST layer
  - Translates simple queries directly to CQP strings
  - Uses closures to pass column parameter (no globals!)
  - ~150 lines of clean, well-documented code

- ✅ **`matcher.py`** - Updated search interface
  - `search()` - New default function using simple query language
  - `search_cqp()` - Renamed from `search()`, for CQP queries
  - Both support `column` parameter (defaults to "token")

- ✅ **`test_simple_query.py`** - Comprehensive test suite
  - 18 tests, all passing
  - Tests verify **actual matched content**, not just counts
  - Helper function `get_matched_tokens()` for assertions

- ✅ **Documentation** - Updated CLAUDE.md
  - Architecture section describes both query languages
  - Usage examples for simple queries
  - Notes about Phase 1 vs Phase 2 features

### Features Implemented (Phase 1)

| Feature | Syntax | Example | Status |
|---------|--------|---------|--------|
| Basic words | `word` | `fox` | ✅ |
| Case-insensitive | default | `the` → The, the | ✅ |
| Wildcards | `?`, `*`, `+` | `s?ng`, `*able`, `+able` | ✅ |
| Alternatives | `[a,b,c]` | `[car,truck]` | ✅ |
| Word sequences | `word1 word2` | `quick brown fox` | ✅ |
| Gap tokens | `*`, `+` | `fox * over`, `fox + over` | ✅ |
| Escaping | `\?`, `\*` | `what\?` | ✅ (untested) |
| Column selection | `column=` | `search(c, "NN*", column="pos")` | ✅ |
| **POS tags** | `word_TAG` | `lights_NN2`, `*ly_AJ0`, `_PNX` | ✅ |
| **Lemma searches** | `{lemma}`, `{lemma/POS}` | `{light}`, `{light/V}` | ✅ |

### Code Quality Improvements Made

During implementation, we refactored several times for cleaner code:

1. **Removed `build_simple_grammar()` wrapper** - Grammar defined at module level
2. **Removed `escape_regex()` wrapper** - Just use `re.escape()` directly
3. **Removed all AST classes** - Build CQP strings directly in parse actions
4. **Removed global `_column` variable** - Pass through closures instead
5. **Improved tests** - Verify actual matched content, not just counts

Final result: ~150 lines of clean, functional code with no unnecessary abstractions.

## What's Done (Phase 2) ✅

### Implementation

- ✅ **POS tag searches** - `word_TAG` and `_TAG` syntax
  - Supports wildcards in both word and POS parts
  - Examples: `lights_NN2`, `*ly_AJ0`, `_PNX`
  - Configurable POS column (default: "pos")

- ✅ **Lemma searches** - `{lemma}`, `{lemma/POS}`, and `{lemma}_TAG` syntax
  - Basic lemma matching: `{sing}` → sing, sang
  - Lemma + simplified POS: `{light/V}`, `{table/N}`
  - Lemma + exact POS tag: `{walk}_VBD`, `{be}_V*`
  - Configurable lemma column (default: "lemma")
  - Simplified POS tag mapping supports both BNC CLAWS-5 and Penn Treebank tagsets

- ✅ **Comprehensive test suite** - 16 new tests (34 total, all passing)
  - 6 tests for POS tag searches
  - 10 tests for lemma searches (including `{lemma}_TAG` syntax)

### POS Tag Mapping

The simplified POS tags are mapped to support both BNC CLAWS-5 and Penn Treebank tagsets:

| Simplified | Meaning | Pattern | Matches |
|-----------|---------|---------|---------|
| V, VERB | Verb | `V.*` | VB, VBD, VBZ, etc. |
| N, SUBST | Noun | `N.*` | NN, NNS, NN1, NN2, etc. |
| A, ADJ | Adjective | `(AJ.*\|JJ.*)` | AJ0, JJ, JJR, JJS, etc. |
| ADV | Adverb | `(AV.*\|RB.*)` | AV0, RB, RBR, etc. |

## What's Done (Phase 3 - Partial) ✅

### Implementation (2025-01-28)

- ✅ **Consecutive gap tokens** - `++`, `***`, `+++**` syntax
  - `++` = exactly 2 tokens
  - `***` = 0-3 tokens
  - `+++**` = 3-5 tokens (3 required + 2 optional)
  - Generalizes to any combination of `+` and `*`
  - Uses negative lookahead regex to distinguish from word wildcards

- ✅ **Regex groups with quantifiers** - `(pattern)?`, `(pattern)+`, `(pattern)*`, `(pattern){n}`, `(pattern){m,n}`
  - Optional groups: `(very)? big`
  - One or more: `(very)+ big`
  - Zero or more: `(red)* car`
  - Exact count: `(quick){2}`
  - Range count: `(brown){1,3}`
  - Groups can contain sequences: `(quick brown)+ fox`
  - Groups can contain gaps: `(fox * over)?`
  - Implemented with pyparsing Forward() for recursive grammar

- ✅ **Comprehensive test suite** - 10 new tests (44 total, all passing)
  - 3 tests for consecutive gap tokens
  - 7 tests for regex groups with quantifiers

### Technical Implementation

**Consecutive Gaps:**
- Pattern: `pp.Regex(r'[+*]+(?![a-zA-Z0-9!@#$%^&=\\\-*+?])')`
- Negative lookahead prevents matching word wildcards like `*able`
- Counts `+` (required) and `*` (optional) to generate CQP range: `[]{m,n}`

**Regex Groups:**
- Uses `pp.Forward()` for recursive grammar (groups can contain groups)
- Base items defined separately from sequence items
- Quantifiers parsed as separate tokens
- Groups generate CQP: `(tokens){quantifier}`

## Known Limitations (Post-Phase 3)

1. **No embedded alternatives in words** - `neighbo[u,]r` doesn't work
   - Workaround: Use full alternatives `[neighbour,neighbor]`
   - To fix: Would need more complex parsing

2. **No proximity operators** - `<<s>>`, `<<3>>` not implemented
   - This is the most complex remaining feature

## Phase 4 Features (Future Work)

### Remaining Features

1. **Proximity operators** (Complex!)
   - Syntax: `<<s>>`, `<<3>>`, `<<5<<`, `>>5>>`
   - Need to:
     - Bidirectional proximity (`<<3>>`)
     - Directional proximity (`<<3<<`, `>>3>>`)
     - Sentence-level proximity (`<<s>>`)
     - Nested constraints: `waste <<s>> (time <<3>> money)`
   - **Challenge**: No direct CQP equivalent - may need custom opcodes

### Implementation Notes for Phase 3

#### Proximity Operators
This is the hardest feature. Options:

1. **Translate to complex CQP disjunctions**
   ```
   day <<3>> night →
   ([word="day"%c] []{0,3} [word="night"%c]) |
   ([word="night"%c] []{0,3} [word="day"%c])
   ```
   - Con: Exponential with multiple constraints

2. **Add custom proximity opcodes to Rust**
   - More efficient
   - Requires Rust changes
   - Best long-term solution

3. **Post-process results**
   - Search for both terms separately
   - Filter results by distance
   - Con: Less efficient

Recommend: Start with approach #1 for basic cases, plan for #2 if needed.

## Testing Strategy for Phase 3

1. Add tests for each new feature to `test_simple_query.py`
2. Use `get_matched_tokens()` helper to verify actual content
3. Test edge cases (e.g., nested groups, chained proximity)
4. Performance tests with larger corpora
5. Compare results with equivalent CQP queries when possible

## Files Modified

### Phase 1
- `python/polars_corpus/simple_parser.py` - New file (~150 lines)
- `python/polars_corpus/matcher.py` - Updated search interface
- `python/tests/test_simple_query.py` - New test file (18 tests)
- `CLAUDE.md` - Updated documentation

### Phase 2
- `python/polars_corpus/simple_parser.py` - Added POS and lemma parsing (~250 lines)
- `python/polars_corpus/matcher.py` - Added pos_column and lemma_column parameters
- `python/tests/test_simple_query.py` - Added 14 new tests (32 total)
- `SIMPLE_QUERY_STATUS.md` - Updated status
- `CLAUDE.md` - Updated with Phase 2 examples

### Phase 3 (Partial)
- `python/polars_corpus/simple_parser.py` - Added consecutive gaps and regex groups (~290 lines)
- `python/tests/test_simple_query.py` - Added 10 new tests (44 total)
- `SIMPLE_QUERY_STATUS.md` - Updated status
- `CLAUDE.md` - Updated with Phase 3 examples

## How to Continue

When ready for Phase 3:

1. Review this document
2. Choose which Phase 3 feature to implement first (recommend regex groups)
3. Update `simple_parser.py` grammar to parse the new syntax
4. Add translation logic to generate appropriate CQP
5. Add tests to verify the feature works correctly
6. Update documentation

The code is well-structured for incremental additions!

## Phase 2 Design Decisions

1. **Column names**: Made configurable with sensible defaults ("pos", "lemma")
2. **Simplified POS tags**: Basic mapping supporting both BNC CLAWS-5 and Penn Treebank
3. **Wildcard support**: Full wildcard support in both word and POS parts of patterns
4. **Parsing strategy**: Used pyparsing Combine to prevent whitespace consumption across pattern boundaries

## Post-Phase 2 Refactoring (2025-01-19)

After completing Phase 2, the code was significantly refactored to reduce complexity:

### Architecture Changes
- **Before**: Parser → Tuples → `_token_to_cqp()` with pattern matching → CQP strings
- **After**: Parser with embedded parse actions → CQP strings directly

### Key Improvements
1. **Eliminated intermediate tuple representation** - Removed `Token` type alias and tuple structures
2. **Eliminated `_token_to_cqp` function** - All conversion logic now in parse actions within `_build_grammar()`
3. **Single pass processing** - Parse and generate CQP in one step
4. **Clearer intent** - Each grammar rule's parse action shows exactly what CQP it generates
5. **No pattern matching overhead** - Direct function calls instead of matching on tuple structure
6. **Python 3.10+ pattern matching** - Was briefly used but then eliminated entirely in favor of parse actions

### Code Structure (Final)
- `_make_constraint()` - Helper to build column constraints
- `_make_token()` - Helper to build token patterns
- `_build_grammar()` - Builds grammar with parse actions that generate CQP directly
- `simple_to_cqp()` - Calls `_build_grammar()` and parses query

### Technical Notes
- Grammar must be rebuilt for each call to `simple_to_cqp()` because parse actions are closures over column names
- This is fast enough for typical usage - pyparsing grammar construction is not expensive
- Alternative considered: Cache grammars by (column, pos_column, lemma_column) tuple - decided unnecessary for now

## Notes for Phase 3

### Before Starting Phase 3
1. Review the current `_build_grammar()` structure in `simple_parser.py`
2. Note that parse actions now generate CQP directly (no intermediate representation)
3. All new features should follow this pattern: grammar rule + parse action that returns CQP string
4. Helper functions `_make_constraint()` and `_make_token()` are available for CQP generation

### Testing Approach
- Continue using `get_matched_tokens()` helper to verify actual matched content
- All 34 existing tests pass and should continue to pass
- Add new tests following existing patterns in `test_simple_query.py`

### Python Version
- Now requires Python 3.10+ (dropped 3.9 support)
- Can use pattern matching if needed, but parse actions are preferred for grammar rules

## References

- `simple_grammar.md` - Full BNF grammar specification
- `Simple_query_language.pdf` - Original BNCweb documentation
- `simple_query_implementation_plan.md` - Initial planning document (may be outdated)
