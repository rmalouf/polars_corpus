# CQP Query Language Grammar

## BNF Specification

```bnf
<query> ::= <disjunction>

<disjunction> ::= <concatenation> ( "|" <concatenation> )*

<concatenation> ::= <repetition>+

<repetition> ::= <primary> "*"
               | <primary> "+"
               | <primary> "?"
               | <primary> "{" <number> "}"
               | <primary> "{" <number>? "," <number>? "}"
               | <primary>

<primary> ::= <node>
            | "(" <query> ")"

<node> ::= "[" <constraint_formula>? "]"

<constraint_formula> ::= <token_disj>

<token_disj> ::= <token_conj> ( "|" <token_conj> )*

<token_conj> ::= <constraint> ( "&" <constraint> )*

<constraint> ::= <atomic_constraint>
               | "(" <constraint_formula> ")"

<atomic_constraint> ::= <feature> <operator> <value>

<operator> ::= "=" | "!="

<feature> ::= [a-zA-Z0-9_]+

<value> ::= '"' [^"]* '"'

<number> ::= [0-9]+
```

## Key Features

1. **Token Constraints**: `[feature="value"]` or `[feature!="value"]` for exact/negated matches
2. **Regular Expressions**: Values are treated as regex patterns (anchored with `^` and `$`)
3. **Boolean Logic**: `&` (AND) and `|` (OR) operators within token constraints
4. **Repetition Operators**:
   - `*` (zero or more)
   - `+` (one or more)  
   - `?` (zero or one)
   - `{n}` (exactly n times)
   - `{m,n}` (between m and n times)
   - `{m,}` (m or more times)
   - `{,n}` (up to n times)
5. **Grouping**: Parentheses for grouping patterns
6. **Disjunction**: `|` for alternative patterns at the top level

## Example Queries

- `[token="word"]` - Match exact token
- `[pos="NOUN"]` - Match by part-of-speech
- `[c5="AJ.*"]+` - One or more adjectives (regex pattern)
- `[token="the"] [pos="NOUN"]` - Sequence matching
- `([pos="ADJ"] | [pos="NOUN"])*` - Zero or more adjectives or nouns
- `[pos="DT"] [c5="AJ.*"]* [pos="NOUN"]` - Determiner followed by optional adjectives and a noun
- `[token="very"] [pos="ADV" | pos="ADJ"]` - "very" followed by adverb or adjective