# CQP Query Language Reference

The concordancer's **CQP Query Language** describes a match as a sequence of
token constraints, each written in square brackets. It supports regular
expressions over any annotation column, boolean combinations of constraints,
sequences, alternation, quantifiers, and variable bindings.

CQP queries are run with `search_cqp()`. The
[Simple query language](simple_query.md) is a shorthand that is translated into
CQP internally, so anything a simple query can express can be written here too.
What CQP adds is the ability to search any column of the corpus, not just the
token, POS and lemma columns, and to say exactly which patterns are
case-sensitive.

```python
import polars_corpus as plc

results = plc.search_cqp(corpus, '[pos="DT"] [pos="JJ"]* [pos="NN"]')
```

## Quick reference

| Query                              | Meaning                                                   |
|------------------------------------|-----------------------------------------------------------|
| `[token="fox"]`                    | the token *fox*                                           |
| `[token="fox"%c]`                  | *fox* in any combination of upper and lower case          |
| `[token="s.ng"]`                   | a token such as *sing*, *sang*, or *song*                 |
| `[token=".*able"]`                 | a token ending in *able*                                  |
| `[pos="NN.*"]`                     | a token whose POS tag begins with `NN`                    |
| `[]`                               | any token at all                                          |
| `[token!="the"]`                   | any token other than *the*                                |
| `[pos="JJ" & lemma="light"]`       | both constraints at once                                  |
| <code>\[pos="DT" &#124; pos="PRP"\]</code> | either constraint                                         |
| `[speaker="A" & lemma="think"]`    | a constraint on any annotation column                     |
| `[pos="DT"] [pos="JJ"] [pos="NN"]` | the three tokens in sequence                              |
| `[token="fox"] [] [token="over"]`  | *fox*, one intervening token, then *over*                 |
| `[token="as"] []{0,3} [token="as"]`| *as*, up to three intervening tokens, then *as*           |
| `[pos="RB"]? [pos="JJ"]`           | an optional adverb, then an adjective                     |
| `[pos="JJ"]+ [pos="NN"]`           | one or more adjectives, then a noun                       |
| `[pos="JJ"]{2,4}`                  | between two and four adjectives                           |
| <code>\[pos="DT"\] \[pos="NN"\] &#124; \[pos="PRP"\]</code> | either the sequence or the pronoun                    |
| `([pos="JJ"] [pos="CC"])+`         | one or more repetitions of the two-token sequence         |
| `$adjs: ([pos="JJ"]+)`             | match the adjectives and bind them to variable `adjs`     |

---

## Token constraints

A query matches a run of tokens, one per bracketed **node**. The simplest node
constrains one annotation column:

```text
[token="fox"]
```

The name before the `=` is a column of the corpus, and the double-quoted value
is a regular expression it must match. The columns a corpus carries are up to
whoever built it, so any of them can be named:

```text
[lemma="be"]
[pos="VB.*"]
[speaker="A"]
[genre="fiction"]
```

An attribute name must begin with a letter or underscore and continue with
letters, digits, or underscores. A column whose name has a space or a hyphen in
it cannot be reached from a query; rename it first. The column must hold
strings.

A node with nothing in it matches any token:

```text
[]
```

### Whole-token matching

The value is matched against the **whole token**, not against part of it. So

```text
[token="ju"]
```

does not match *jumps*, and

```text
[token=".*ing"]
```

is how to ask for tokens ending in *ing*. The pattern is wrapped in `^(...)$`
before it is run, so anchors of your own are unnecessary.

### Case

Matching is **case-sensitive** by default. `[token="the"]` matches *the* but not
*The*. Adding `%c` after the closing quote folds case for that one constraint:

```text
[token="the"%c]
```

The flag is Unicode-aware, so `[token="CAFÉ"%c]` matches *café*. It applies to
the constraint it follows, not to the whole node, so this is case-sensitive in
the tag and case-insensitive in the word form:

```text
[pos="NN" & token="Brown"%c]
```

`%c` is the only modifier the language accepts.

### Negation

`!=` requires the value **not** to match:

```text
[token!="the"]
[pos!="NN.*"]
```

A token whose column is null matches neither `=` nor `!=`; only `[]` matches it.

### Combining constraints

`&` is *and*, `|` is *or*, and `&` binds more tightly than `|`. Parentheses
group them:

```text
[pos="JJ" & lemma="brown"]
[pos="DT" | pos="PRP"]
[pos="JJ" & (lemma="quick" | lemma="lazy")]
```

There is no `!` prefix for a whole constraint; negate the individual comparisons
with `!=` instead. `[pos!="JJ" & pos!="DT"]` is a token that is neither an
adjective nor a determiner.

---

## Regular expressions

Values are regular expressions in the Rust `regex` syntax, which polars uses.
The common constructs are all available:

| Pattern      | Matches                                            |
|--------------|----------------------------------------------------|
| `.`          | any one character                                  |
| `.*`, `.+`   | any run of characters, empty or not                |
| `[A-Z]`      | one character from a class                          |
| `\w`, `\d`, `\s` | a word, digit, or space character              |
| `\p{Lu}`     | one uppercase letter, in any script                |
| <code>(a&#124;b)</code> | either alternative                                 |
| `x?`, `x{2,3}` | quantified characters                            |

For example:

```text
[token="\w+ing"]
[token="[Tt]he"]
[lemma="(go|come)"]
[pos="(VB|VBD|VBN)"]
```

Look-around (`(?<=...)`, `(?=...)`) and backreferences are not part of this
regex flavor and raise an error. An inline flag group works, so
`[token="(?i)fox"]` is another way to write `[token="fox"%c]`.

### Escaping

Regex metacharacters need a backslash to be taken literally:

```text
[token="U\.S\."]
[pos="\("]
```

`$` is one of them, which matters for the Penn Treebank tags that end in it.
`[pos="PRP$"]` matches nothing, because the `$` is read as an end-of-string
anchor. The tag is written:

```text
[pos="PRP\$"]
```

The value is delimited by double quotes, so a literal `"` is written `\"` and a
literal backslash `\\`:

```text
[token="say \"hi\""]
[token="back\\slash"]
```

Nothing else needs escaping. Apostrophes, `%`, and non-ASCII letters can be
written directly, so `[token="don't"]`, `[token="50%"]`, and `[token="café"]`
are all queries in their own right. Single quotes cannot delimit a value.

---

## Sequences

Writing nodes one after another matches them in sequence:

```text
[pos="DT"] [pos="JJ"] [pos="NN"]
```

Whitespace between nodes is not significant, and a long query can be broken
across lines:

```text
[pos="DT"]
[pos="JJ"]
[pos="NN"]
```

An empty node is how a gap of fixed width is written:

```text
[token="fox"] [] [token="over"]
```

matches *fox*, one intervening token of any kind, then *over*.

### Alternation

`|` between sequences matches either of them:

```text
[token="fox"] | [token="dog"]
```

It binds more loosely than sequence, so

```text
[pos="JJ"] [pos="NN"] | [pos="DT"] [pos="JJ"]
```

is two two-token alternatives rather than a four-token sequence with something
optional in the middle. Parentheses group an alternation into a larger
sequence:

```text
[pos="DT"] ([pos="JJ"] [pos="NN"] | [pos="NN"])
```

### Quantifiers

A quantifier follows the node or parenthesized group it repeats:

| Quantifier | Repetitions            |
|------------|------------------------|
| `?`        | zero or one            |
| `*`        | zero or more           |
| `+`        | one or more            |
| `{n}`      | exactly *n*            |
| `{m,n}`    | between *m* and *n*    |
| `{m,}`     | *m* or more            |
| `{,n}`     | up to *n*              |
| `{,}`      | any number, as for `*` |

```text
[pos="JJ"]* [pos="NN"]
[pos="DT"]? [pos="JJ"] [pos="NN"]
[token="as"] []{0,3} [token="as"]
([pos="JJ"] [pos="CC"])+ [pos="JJ"]
```

`{m,n}` with `m` greater than `n` is an error.

---

## How a match is found

The matcher walks the corpus from the beginning, trying each token position in
turn. Three rules decide what comes back:

**The longest match wins.** Where a query can match more than one way starting
at the same token, the longest of them is reported. Quantifiers are therefore
greedy, and alternation takes the longest branch rather than the first one
written. Over *the big red house*, this query reports the whole four-token
phrase, not the three-token one the first branch allows:

```text
[pos="DT"] ([pos="JJ"] | [pos="JJ"] [pos="JJ"]) [pos="NN"]
```

**Matches never overlap.** Once a match is found the scan resumes at the first
token past it, so `[]{2}` over a nine-token file reports four matches, not
eight.

**A match never crosses a file boundary.** The `file_id_column` argument of
`search_cqp()` names the column that marks where one text ends and the next
begins, and no match spans a change in its value. Pass `None` to search the
corpus as one continuous run of tokens.

A match must cover at least one token. A query that can only ever match zero
tokens, such as `[pos="XX"]?` where no token is tagged `XX`, reports nothing;
`search_cqp()` returns `None` when a query matches nowhere.

---

## Variable bindings

`$name:` binds what the following node or group matched, so that part of the
match can be read back on its own:

```text
$noun: [pos="NN"]
```

A binding takes **one node or one parenthesized group**. That distinction
matters as soon as a quantifier is involved:

```text
$x: [pos="JJ"]+
```

repeats the binding rather than binding the repetition, and `x` ends up holding
only the last adjective. Parenthesize what is to be bound:

```text
$x: ([pos="JJ"]+)
```

Several bindings can appear in one query, and they may be nested:

```text
$det: [pos="DT"] $adjs: ([pos="JJ"]*) $head: [pos="NN"]
$np: ([pos="DT"] $head: ([pos="NN"]))
```

Variable names begin with an ASCII letter or underscore and continue with
letters, digits, or underscores. A name may be used only once in a query;
reusing one raises an error.

### Reading the bindings back

A concordance gives each bound variable a column of its own, named after the
column it was taken from:

```python
results = plc.search_cqp(corpus, '$det: [pos="DT"] $adj: [pos="JJ"] $noun: [pos="NN"]')
results.variables                       # ['det', 'adj', 'noun']
conc = results.concordance("token", window=5)
conc.columns
# ['token_left_context', 'token', 'token_right_context',
#  'token_det', 'token_adj', 'token_noun']
```

Each holds the tokens the variable captured, so the usual list expressions group
and count them:

```python
conc.group_by(pl.col("token_adj").list.join(" ")).len().sort("len", descending=True)
```

`bindings=` chooses which to include -- a name, a list of names, or `False` for
none:

```python
results.concordance("token", window=5, bindings="noun")
```

A variable is null on the lines whose match never bound it, as an optional
subpattern's can be, and an empty list where it matched no token at all.

---

## Relation to the Simple query language

`search()` translates a simple query into CQP and hands it to the same matcher
`search_cqp()` uses. The translation shows how the pieces line up:

| Simple query    | CQP query                                     |
|-----------------|-----------------------------------------------|
| `fox`           | `[token="fox"%c]`                             |
| `s?ng`          | `[token="s.ng"%c]`                            |
| `*able`         | `[token=".*able"%c]`                          |
| `[car,truck]`   | <code>\[token="(?:car&#124;truck)"%c\]</code> |
| `lights_NN2`    | `[token="lights"%c & pos="NN2"%c]`            |
| `_{SUBST}`      | `[pos="N.*"%c]`                               |
| `{light/V}`     | `[lemma="light"%c & pos="V.*"%c]`             |
| `fox + over`    | `[token="fox"%c] []{1} [token="over"%c]`      |
| `fox +* over`   | `[token="fox"%c] []{1,2} [token="over"%c]`    |
| `$x: fox`       | `$x: ([token="fox"%c])`                       |

Two differences are worth keeping in mind when moving from one to the other.
A simple query is case-insensitive throughout -- every constraint it emits
carries `%c` -- while a CQP query is case-sensitive unless it says otherwise.
And a simple query reaches its three columns through the `token_column`,
`pos_column` and `lemma_column` arguments of `search()`, whereas a CQP query
names its columns itself and `search_cqp()` has no such arguments.

---

## What this dialect leaves out

The language here is the token-level core of CQP. It has no default attribute,
so a bare `"fox"` is an error rather than a search of the token column, and it
does not support structural attributes (`<s> ... </s>`), region operators
(`within`, `containing`), global constraints (`::`), target markers (`@`), the
`MU`/`meet`/`union` query form, or any modifier but `%c`.

---

## Query structure

```text
query        := disjunction

disjunction  := sequence ("|" sequence)*

sequence     := repetition+

repetition   := primary quantifier?

primary      := node
              | "(" query ")"
              | binding

binding      := "$" NAME ":" (node | "(" query ")")

node         := "[" formula? "]"

formula      := conjunction ("|" conjunction)*

conjunction  := constraint ("&" constraint)*

constraint   := atomic
              | "(" formula ")"

atomic       := NAME ("=" | "!=") '"' REGEX '"' "%c"?

quantifier   := "?"
              | "+"
              | "*"
              | "{" INTEGER "}"
              | "{" INTEGER "," INTEGER "}"
              | "{" INTEGER "," "}"
              | "{" "," INTEGER "}"
              | "{" "," "}"
```

An empty query is not valid.
