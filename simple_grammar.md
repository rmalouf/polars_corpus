# Simple Query Language Grammar (BNCweb Style - Core Features)

## BNF Specification

```bnf
query ::= sequence_item+

sequence_item ::= word_token
               | gap_token
               | proximity_expr
               | regex_group

word_token ::= simple_word
            | wildcard_word
            | alternative_list
            | escaped_char

simple_word ::= ( letter | digit | [!@#$%^&_=\-] )+

wildcard_word ::= wildcard_pattern

wildcard_pattern ::= ( letter | digit | [!@#$%^&_=\-] | wildcard )+

wildcard ::= "?" | "*" | "+"

alternative_list ::= "[" alternative_sequence "]"

alternative_sequence ::= alternative ( "," alternative )*

alternative ::= wildcard_word

escaped_char ::= "\" metacharacter

metacharacter ::= "?" | "*" | "+" | "," | ":" | "@" | "/" | "(" | ")" 
               | "[" | "]" | "{" | "}" | "_" | "-" | "<" | ">"

gap_token ::= "+" | "*"

proximity_expr ::= target_item proximity_op constraint_item

target_item ::= word_token

constraint_item ::= word_token | "(" proximity_expr ")"

proximity_op ::= "<<" distance? ">>"
              | ">>" distance ">>"
              | "<<" distance "<<"

distance ::= "s" | digit+

regex_group ::= "(" regex_content ")"

regex_content ::= regex_item ( "|" regex_item )*

regex_item ::= sequence_item quantifier?

quantifier ::= "?"
            | "*"
            | "+"
            | "{" digit+ "}"
            | "{" digit+ "," digit+ "}"

modifier ::= ":d"

letter ::= [a-zA-Z]
digit ::= [0-9]
```

## Language Features

### Basic Word Searches
- **Simple words**: `glitterati` matches exact word forms
- **Wildcards**:
  - `?`: single character (`s?ng` → sing, sang, song)
  - `*`: zero or more characters (`*able` → able, table, capable)
  - `+`: one or more characters (`+able` → table, capable, but not able)
  - Combined: `??+able` (3+ chars before "able")

### Alternatives
- **Square brackets**: `[able,ability]`, `neighbo[u,]r`
- Can include wildcards: `??+[able,ability]`

### Word Sequences
- **Multiple words**: `talk of the town`
- **Gaps**:
  - `*`: optional token (`eat * up` → eat up, eat it up)
  - `+`: required gap (`eat + up` → eat it up, but not eat up)
  - Multiple: `eat +++** up` (skip 3-5 tokens)

### Regular Expression Patterns
- **Quantifiers**: `?`, `*`, `+`, `{2,4}`
- **Alternatives**: `(very | quite | rather)`
- **Grouping**: `(very)?` (optional word)

### Proximity Queries
- **Same sentence**: `kick <<s>> bucket`
- **Token distance**: `day <<3>> night` (within 3 tokens)
- **Directional**: `day <<5<< night`, `day >>5>> night`
- **Chained**: `day <<5>> month <<5>> year`

### Modifiers
- **Case sensitivity**: Default case-insensitive
- **Accent insensitive**: `:d` modifier (`fiancee:d`)

### Escaping
- **Backslash**: `\?` for literal question mark
- **Metacharacters**: `? * + , : @ / ( ) [ ] { } _ - < >`

