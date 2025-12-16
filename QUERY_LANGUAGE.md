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
plc.search(c, '(very)+ big')          # One or more
plc.search(c, '(red){2,3}')           # 2-3 repetitions
```

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