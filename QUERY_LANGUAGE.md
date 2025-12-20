# Query Language Reference

## Simple Query Language (Default)
Case-insensitive, user-friendly syntax:
```python
# Basic patterns
plc.search(c, 'quick brown fox')      # Word sequence
plc.search(c, '*able')                # Wildcards: ? (one), * (zero+), + (one+)
plc.search(c, '[car,truck]')          # Alternatives

# With linguistic annotations
plc.search(c, 'word_TAG')             # Word + POS tag
plc.search(c, '_TAG')                 # POS tag only
plc.search(c, '{light}')              # Lemma (all forms)
plc.search(c, '{light/V}')            # Lemma + simplified POS
plc.search(c, '{walk}_VBD')           # Lemma + exact POS

# Gaps and sequences
plc.search(c, 'fox * over')           # Optional gap (0-1 tokens)
plc.search(c, 'fox + over')           # Required gap (1 token)
plc.search(c, 'fox ++ over')          # Exactly 2 tokens
plc.search(c, 'fox *** over')         # 0-3 tokens

# Repetition groups
plc.search(c, '(very)? big')          # Optional
plc.search(c, '(very)+ big')          # One or more (no space before +)
plc.search(c, '(very) + big')         # (very) followed by gap token, then big
plc.search(c, '(red){2,3}')           # 2-3 repetitions

**Note:** Whitespace matters! `(pattern)+` (no space) is a quantifier meaning "one or more repetitions",
while `(pattern) +` (with space) means "pattern followed by exactly one mandatory token".
```

### Variable Bindings (Capturing Subpatterns)
Capture parts of matches using `$varname: pattern` syntax:
```python
# Basic binding - capture single word
plc.search(c, '$target: fox')         # Capture "fox"
results._matches[0].bindings['target']  # → Span(3, 4)

# Multiple variables
plc.search(c, '$det: the $noun: fox')  # Capture both "the" and "fox"

# Bind patterns with wildcards
plc.search(c, '$suffix: *able')        # Capture words ending in 'able'

# Bind linguistic features
plc.search(c, '$verb: _VBD')           # Capture past tense verbs
plc.search(c, '$lemma: {walk}')        # Capture any form of "walk"

# Bind groups for multi-token spans
plc.search(c, '$phrase: (quick brown) fox')  # Captures "quick brown"
plc.search(c, '($mods: very)+ big')          # Captures "very" or "very very"

# Bind alternatives
plc.search(c, '$vehicle: [car,truck]')       # Captures whichever matches
```

**Accessing bindings:**
```python
results = plc.search(corpus, '$det: the $noun: fox')
for match in results._matches:
    det_span = match.bindings['det']
    noun_span = match.bindings['noun']
    det_text = corpus['token'][det_span.start:det_span.end]
    noun_text = corpus['token'][noun_span.start:noun_span.end]
```

**Notes:**
- Variable names must start with a letter, contain only letters/numbers/underscores
- Parentheses are optional: `$x: fox` and `$x: (fox)` both work
- Same syntax as CQP queries for easy transition

## CQP Query Language
Advanced linguistic patterns:
```python
plc.search_cqp(c, '[c5="AJ.*"]+ [c5="NN.*"]+')  # Adj+ followed by Noun+
plc.search_cqp(c, '[token="the"] [pos="NN.*"]')
```

### Variable Bindings
Capture subpatterns using `$varname: (pattern)` syntax:
```python
# Basic binding
plc.search_cqp(c, '$n: ([pos="NN"])')  # Capture noun

# Multiple variables
plc.search_cqp(c, '$det: ([pos="DT"]) $adj: ([pos="JJ"]) $noun: ([pos="NN"])')

# With quantifiers
plc.search_cqp(c, '$adjs: ([pos="JJ"]+) [pos="NN"]')  # Capture all consecutive adjectives

# Optional patterns
plc.search_cqp(c, '$det: ([pos="DT"]?) [pos="JJ"] [pos="NN"]')  # May bind empty span

# With alternation
plc.search_cqp(c, '$target: ([pos="JJ"] | [pos="NN"])')  # Works with backtracking

# Nested bindings
plc.search_cqp(c, '$phrase: (($det: ([pos="DT"])) [pos="JJ"] [pos="NN"])')
```

**Important notes:**
- Parentheses are **required** around bound patterns: `$var: (pattern)`
- Variable names must be unique within a query
- Bindings are accessible via `match.bindings[varname]` → `Span(start, end)`
- Empty spans (start==end) for zero-width matches (e.g., optional patterns that didn't match)

## Translation
Simple queries are translated to CQP before matching, ensuring consistent behavior.