# CQP Query Language

CQP (Corpus Query Processor) syntax provides precise token-level pattern matching using
attribute constraints and regular expression operators.

## Basic Token Constraints

A token is written as `[feature="value"]`. Values are regular expressions anchored at both ends.

```python
plc.search_cqp(c, '[token="the"]')           # Exact token match
plc.search_cqp(c, '[token="wom.n"]')         # Regex: woman or women
plc.search_cqp(c, '[token="walk.*"]')        # Regex: walk, walks, walked, walking
plc.search_cqp(c, '[token="walk.*"%c]')      # Case-insensitive match
plc.search_cqp(c, '[pos="NN.*"]')            # Any singular/plural noun
plc.search_cqp(c, '[token!="the"]')          # Negation: anything but "the"
[]                                           # Any single token (wildcard)
```

## Boolean Operators Within Tokens

Use `&` (AND) and `|` (OR) to combine constraints on a single token:

```python
plc.search_cqp(c, '[lemma="house" & pos="NN"]')           # Lemma AND POS
plc.search_cqp(c, '[pos="JJ" | pos="JJR" | pos="JJS"]')  # Any adjective form
plc.search_cqp(c, '[pos="NN" & token!="man"]')            # Noun, not "man"
plc.search_cqp(c, '[(pos="JJ" | pos="RB") & token="very"]') # Grouped conditions
```

## Sequences

Space-separate tokens to match sequences:

```python
plc.search_cqp(c, '[token="the"] [pos="NN.*"]')           # "the" + noun
plc.search_cqp(c, '[c5="AJ.*"]+ [c5="NN.*"]+')           # Adj+ followed by Noun+
plc.search_cqp(c, '[pos="DT"] [c5="AJ.*"]* [pos="NN"]')  # Det + optional Adj* + Noun
```

## Repetition Operators

```python
plc.search_cqp(c, '[pos="JJ"]*')       # Zero or more adjectives
plc.search_cqp(c, '[pos="JJ"]+')       # One or more adjectives
plc.search_cqp(c, '[pos="JJ"]?')       # Optional adjective
plc.search_cqp(c, '[pos="JJ"]{2}')     # Exactly 2 adjectives
plc.search_cqp(c, '[pos="JJ"]{2,4}')   # 2 to 4 adjectives
plc.search_cqp(c, '[pos="JJ"]{2,}')    # 2 or more adjectives
plc.search_cqp(c, '[pos="JJ"]{,4}')    # Up to 4 adjectives
```

## Grouping and Disjunction

Use parentheses to group patterns, and `|` for top-level alternation:

```python
plc.search_cqp(c, '([pos="ADJ"] | [pos="ADV"])* [pos="NOUN"]')
plc.search_cqp(c, '[token="very"] ([pos="ADV"] | [pos="ADJ"])')
plc.search_cqp(c, '([token="not"] [token="only"] | [token="not"] [token="just"])')
```

## Variable Bindings

Capture subpatterns using `$varname: (pattern)`. Parentheses around the bound pattern are required.

```python
# Bare node binding (simple case)
plc.search_cqp(c, '$n: [pos="NN"]')

# Multiple variables
plc.search_cqp(c, '$det: [pos="DT"] $adj: [pos="JJ"] $noun: [pos="NN"]')

# Use parentheses when the bound pattern spans multiple tokens or has a quantifier
plc.search_cqp(c, '$adjs: ([pos="JJ"]+) [pos="NN"]')

# Optional — binds an empty span if the pattern doesn't match
plc.search_cqp(c, '$det: ([pos="DT"]?) [pos="JJ"] [pos="NN"]')

# Alternation
plc.search_cqp(c, '$target: ([pos="JJ"] | [pos="NN"])')

# Nested bindings
plc.search_cqp(c, '$phrase: (($det: [pos="DT"]) [pos="JJ"] [pos="NN"])')
```

**Accessing bindings:**

```python
results = plc.search_cqp(corpus, '$det: [pos="DT"] $noun: [pos="NN"]')
for match in results._matches:
    det_span = match.bindings['det']    # Span(start, end)
    noun_span = match.bindings['noun']
```

**Notes:**
- For a single token, `$var: [feature="value"]` works directly
- For multi-token or quantified patterns, parentheses are required: `$var: (pattern+)`
- Variable names must be unique within a query
- Empty spans (`start == end`) are returned for optional patterns that didn't match

## Full Grammar Reference

See [CQP grammar BNF](../../cqp_grammar.md) for the complete formal specification.
