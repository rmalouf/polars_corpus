# Simple Query Language Reference

The concordancer's **Simple Query Language** provides a compact syntax for common corpus searches. It supports word forms, wildcards, lemmas, part-of-speech tags, alternatives, gaps, grouping, repetition, and variable bindings.

Simple queries are translated internally into CQP expressions. In most cases you do not need to know CQP to use the language.

## Quick reference

| Query                    | Meaning                                                         |
|--------------------------|-----------------------------------------------------------------|
| `fox`                    | the word *fox*                                                  |
| `s?ng`                   | a word such as *sing*, *sang*, or *song*                        |
| `*able`                  | a word ending in *able*                                         |
| `walk+`                  | a word beginning with *walk* followed by one or more characters |
| `*[able,ability]`        | any word ending with *able* or *ability*                        |
| `neighbo[u,]r`           | either *neighbour* or *neighbor*                                |
| `quick brown fox`        | the three words in sequence                                     |
| `fox + over`             | *fox*, one intervening token, then *over*                       |
| `fox * over`             | *fox*, zero or one intervening token, then *over*               |
| `as *** as`              | *as*, up to three intervening tokens, then *as*                 |
| `as +* as`               | *as*, one or two intervening tokens, then *as*                  |
| `lights_NN2`             | *lights* with POS tag `NN2`                                     |
| `_NN2`                   | any token with POS tag `NN2`                                    |
| `fox_{N}`                | *fox* with a noun tag                                           |
| `the _{N}`               | *the* followed by any noun                                      |
| `[the,a] _{N}`           | *the* or *a* followed by any noun                               |
| `New\_York_NNP`          | the word *New_York* with POS tag `NNP`                          |
| `{light}`                | a token whose lemma is *light*                                  |
| `{light/V}`              | lemma *light* with a verb tag                                   |
| `{walk}_VBD`             | lemma *walk* with POS tag `VBD`                                 |
| `{box}_{SUBST}`          | lemma *box* with a noun tag                                     |
| `(cat\|dog)`             | either *cat* or *dog*                                           |
| `(red fox\|blue whale)`  | either of the two sequences                                     |
| `(very)? good`           | optional *very*, followed by *good*                             |
| `(ha){2,4}`              | between two and four occurrences of *ha*                        |
| `$x: fox`                | match *fox* and bind it to variable `x`                         |
| `$phrase: (quick brown)` | bind the sequence *quick brown* to `phrase`                     |

---

## Tokens

A plain word matches a token with that word form:

```text
fox
```

Multiple expressions separated by whitespace form a sequence:

```text
quick brown fox
```

This matches three consecutive tokens.

Whitespace between query elements is not significant. Spaces, tabs, and line breaks may all be used to separate elements.

Word-form searches are **case-insensitive**. For example:

```text
fox
```

matches *fox*, *Fox*, or *FOX* or any combination of upper and lower case.

A word form may contain any character, so `don't`, `U.S.`, and `café` are all queries in their own right. The few characters that carry syntax are written with a backslash before them; see [Literal characters and escaping](#literal-characters-and-escaping).

### Character wildcards

Three wildcards can be used inside word forms:

| Wildcard | Meaning                 |
| -------- | ----------------------- |
| `?`      | exactly one character   |
| `*`      | zero or more characters |
| `+`      | one or more characters  |

For example:

```text
s?ng
```

matches four-character forms with any second character, such as *sing*, *sang*, and *song*.

```text
*able
```

matches any token ending in *able*.

```text
walk+
```

matches a token beginning with *walk* and containing at least one additional character.

Wildcards operate on **characters within a token**. This is different from a `+` or `*` appearing by itself, which denotes a gap between tokens; see [Token Sequences](#gap)

### Alternatives

Square brackets list comma-separated alternatives for one part of a word form:

```text
[car,truck]
```

matches either *car* or *truck*.

Bracketed list can be combined with other characters and with wildcards:

```text
??+[able,ability]
```

matches *capable*, *capability*, *availability*, … — any token with at least three characters before *able* or *ability*.

An alternative may be empty, which makes the rest of the group optional:

```text
neighbo[u,]r
```

matches both *neighbour* and *neighbor*.

The alternatives themselves may contain wildcards:

```text
[walk*,run*]
```

matches a token satisfying either pattern.

A word may contain more than one bracketed list, and a list may also appear in the word part or the tag part of a POS constraint:

```text
walk[s,ed]_V*
```

Whitespace around an alternative is ignored, so these are equivalent:

```text
[cat,dog]
[cat, dog]
```

### Part-of-speech constraints

A POS constraint is introduced by an underscore:

```text
word_TAG
```

For example:

```text
lights_NN2
```

matches the word form *lights* only when its POS tag is `NN2`.

Omit the word form to search only by POS:

```text
_NN2
```

This matches any token whose POS tag is `NN2`.

Wildcards may also be used in POS tags:

```text
_V*
```

matches POS tags beginning with `V`.

Likewise, word-form wildcards and POS constraints can be combined:

```text
walk*_V*
```

Several generic POS categories are recognized and expanded to patterns appropriate for BNC CLAWS-5 and Penn Treebank-style tagsets. A category name must be written **in braces**:

```text
fox_{SUBST}
```

Without them the name is an ordinary tag pattern, so `fox_N` looks for the literal tag `N`. This keeps every tag reachable in corpora tagged with a scheme that spells its tags the way these categories are spelled, such as Universal Dependencies, where `_ADJ` and `_PRON` are tags in their own right.

The available names are:

| Name   | Also accepted | POS pattern           |
| ------ | ------------- | --------------------- |
| `V`    | `VERB`        | `V.*`                 |
| `N`    | `SUBST`       | `N.*`                 |
| `A`    | `ADJ`         | `AJ.*` or `JJ.*`      |
| `ADV`  | —             | `AV.*` or `RB.*`      |
| `ART`  | —             | `AT.*` or `DT`        |
| `CONJ` | —             | `CJ.*` or `CC`        |
| `PREP` | —             | `PR.*`, `IN`, or `TO` |
| `PRON` | —             | `PN.*` or `PRP.*`     |
| `INT`  | `INTERJ`      | `ITJ` or `UH`         |
| `STOP` | —             | `PU.*`                |
| `UNC`  | —             | `UNC`                 |

For example:

```text
_{N}
```

matches a token with a POS tag beginning with `N`, and

```text
_{ADJ}
```

matches tags beginning with `AJ` or `JJ`.

A braced name must be one of the names in the table; anything else is an error.

### Lemmas

Curly braces specify a lemma:

```text
{light}
```

This matches tokens whose lemma is *light*, regardless of their surface form.

Lemma searches may contain the same `?`, `*`, and `+` character wildcards and `[a,b]` alternatives used for word forms:

```text
{walk*}
{[car,truck]}
```

Lemma matching is case-insensitive.

A generic word class can be included inside the braces after `/`:

```text
{light/V}
```

This matches lemma *light* with a verb POS tag, using the mappings in the preceding section:

```text
{box/SUBST}
```

matches lemma *box* with a noun tag.

The name after `/` must be one of the generic categories; anything else is an error naming the ones that are available. A literal tag goes after the closing brace instead, so `{word}_NN` is how to ask for the tag `NN` and `{word}_NN*` for any tag beginning with it. Keeping the two apart means `{light/ADJ}` always means the adjective class, never a tagset whose adjective tag happens to be spelled `ADJ`.

A lemma can also be followed by an underscore and POS specification:

```text
{walk}_VBD
```

This requires both:

* lemma `walk`
* POS tag `VBD`

Wildcards are allowed in the external POS specification:

```text
{be}_V*
```

and generic POS names may be used in braces:

```text
{box}_{SUBST}
```

The external form is useful when an exact tag or wildcarded tag pattern is required.

Do not rely on combining both POS notations in the same expression, such as:

```text
{walk/V}_NN
```

When an external `_TAG` is present, the implementation uses that tag constraint and ignores the POS portion inside the lemma braces.

---

## Token sequences

A sequence consisting only of `+` and `*` characters denotes a variable-length gap between query elements.

In gap syntax:

* each `+` requires one intervening token;
* each `*` permits one additional optional token.

For example:

```text
fox + over
```

matches *fox*, followed by exactly one intervening token, followed by *over*.

```text
fox * over
```

matches *fox* followed by zero or one intervening token and then *over*.

```text
fox ++ over
```

requires exactly two intervening tokens.

```text
fox +* over
```

allows one or two intervening tokens.

```text
fox *** over
```

allows between zero and three intervening tokens.

More generally, the minimum gap length is the number of `+` characters, and the maximum gap length is the total number of `+` and `*` characters.

The order of `+` and `*` within a gap does not affect these bounds. For example, `+*` and `*+` both describe a gap of one or two tokens.

<a name="gap"></a>
Whether `+` and `*` are interpreted as character wildcards or gap operators depends on context:

```text
walk*
```

is a **single-token word pattern**.

```text
walk * quickly
```

contains a **zero-or-one-token gap** between *walk* and *quickly*.

### Groups and alternatives

Parentheses group one or more query elements:

```text
(quick brown)
```

Groups are especially useful for alternatives and repetition.

Use `|` inside a group to specify alternatives:

```text
(cat|dog)
```

matches either *cat* or *dog*.

Each branch may contain a complete sequence:

```text
(red fox|blue whale)
```

matches either:

```text
red fox
```

or:

```text
blue whale
```

Alternatives can themselves contain any valid query items, including nested groups.

### Repetition and optional groups

A parenthesized group may be followed immediately by a repetition operator.

- Optional expressions
  ```text
  (very)? good
  ```
   matches either *good* or *very good*.

- Zero or more repetitions
  ```text
  (very)* good
  ```
  allows any number of repetitions of *very* before *good*.

- One or more repetitions
  ```text
  (ha)+
  ```
  matches *ha*, *ha ha*, *ha ha ha*, *ha ha ha ha*, and so on

- Exact repetition
  ```text
  (ha){3}
  ```
  requires exactly three occurrences: *ha ha ha*

- Bounded repetition
  ```text
  (ha){2,4}
  ```
  requires between two and four occurrences: *ha ha*, *ha ha ha*, or *ha ha ha ha*

allows between two and four occurrences.

A repeated group may contain a complete sequence:
```text
(very good){2}
```
matches the two-token sequence twice: *very good very good*

---

## Variable bindings

A matched expression can be assigned to a CQP variable using:

```text
$variable: expression
```

For example:

```text
$x: fox
```

binds the matched *fox* token to the variable `x`.

Several bindings can occur in one query:

```text
$det: the $noun: fox
```

Variable names:

* must begin with an ASCII letter;
* may subsequently contain letters, digits, or `_`.

Examples of valid names include:

```text
$x:
$word:
$noun1:
$target_word:
```

### Binding a sequence

A binding applies only to the **single atom or group immediately following it**.

Thus:

```text
$x: quick brown
```

binds only *quick* to `x`; *brown* remains outside the binding.

To bind an entire sequence, group it:

```text
$x: (quick brown)
```

This binds the complete two-token sequence.

Bindings may likewise be applied to alternatives or repeated groups:

```text
$x: (cat|dog)
```

---

## Literal characters and escaping

Any character can appear in a word form. Punctuation and letters outside ASCII need nothing special:

```text
don't
U.S.
café
```

A handful of characters carry syntax, and a backslash makes one of them literal instead:

```text
? * + , : $ / ( ) | [ ] { } _ < > \
```

together with the space that separates one query item from the next.

So a literal colon within a word is written:

```text
foo\:bar
```

and an underscore must be escaped because an unescaped one introduces a POS constraint:

```text
New\_York_NNP
```

This searches for word form `New_York` with POS tag `NNP`.

A backslash always begins an escape, so a word containing a backslash doubles it (`a\\b`), and a backslash before a letter or a digit is an error rather than a literal.

### Escaped wildcards

An escaped `?`, `*`, or `+` is a literal character rather than a wildcard:

```text
\*able
```

matches the token `*able`, while the unescaped `*able` matches `able`, `table`, `capable`, and so on.

Escapes work the same way inside alternatives and lemma constraints, so `[x\*x,a\,b]` matches the tokens `x*x` and `a,b`, and `{a\/b}` searches for the lemma `a/b` rather than lemma `a` with POS tag `b`.

### Tags made of punctuation

The same rules apply to a POS tag, which matters for the Penn Treebank tags that are punctuation marks. `_.` and `_''` can be written directly, while the tags `:`, `,`, `$`, and `(` need a backslash:

```text
_\:
```

---

## Query structure

At a high level, a query consists of one or more items:

```text
query       := item+

item        := binding
             | group
             | atom

binding     := "$" NAME ":" (group | atom)

group       := "(" sequence ("|" sequence)* ")" quantifier?

quantifier  := "?"
             | "+"
             | "*"
             | "{" INTEGER "}"
             | "{" INTEGER "," INTEGER "}"

atom        := word
             | gap
             | pos_constraint
             | lemma
             | lemma_pos_constraint
```

Adjacent items form a sequence.

An empty query is not valid.
