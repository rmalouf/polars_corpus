# Simple Query Language Implementation Plan

## Overview

Implement a parser for the BNCweb-style Simple Query Language that compiles to the same matcher opcodes as the existing CQP parser, enabling reuse of the matching infrastructure.

## Current Architecture Analysis

### Existing Components

- **Opcodes**: `TOKEN`, `JUMP`, `SPLIT`, `SKIP`, `MATCH`
- **CQP Parser**: Uses pyparsing to build polars expressions and opcode sequences
- **OpcodeMatcher**: Rust-based matcher that executes opcodes against corpus
- **Mask Computation**: Polars-based prefiltering system

### Key Files

- `matcher.py`: Main search interface and mask computation
- `cqp_parser.py`: CQP pyparsing grammar and compilation
- `_internal.pyi`: Rust opcode types and matcher interface

## Implementation Strategy

### Approach 1: Direct Compilation to Opcodes

**Concept**: Parse simple query language directly to opcodes, bypassing CQP altogether.

**Pros**:
- Clean separation of concerns
- Can optimize for simple query patterns
- No dependency on CQP expression building

**Cons**:
- Need to reimplement opcode generation logic
- More complex proximity operator implementation
- Duplicate compilation patterns

### Approach 2: Translation to CQP

**Concept**: Parse simple queries and translate to equivalent CQP expressions, then use existing CQP parser.

**Pros**:
- Reuses all existing CQP compilation logic
- Automatic support for all CQP features
- Easier to maintain consistency

**Cons**:
- Additional translation layer
- May generate suboptimal CQP expressions
- Harder to provide good error messages

### Approach 3: Shared Compilation Infrastructure

**Concept**: Extract common compilation utilities and build both parsers on shared foundation.

**Pros**:
- Best of both worlds
- Cleaner architecture long-term
- Easier to add new query languages

**Cons**:
- Requires refactoring existing code
- More complex initial implementation

## Feature Mapping Challenges

### Wildcard Translation

**Simple Query**: `"fo?"` (single character wildcard)
**Problem**: Need to convert to regex pattern while handling escaping

**Potential Solutions**:
1. Convert to regex: `"fo."` 
2. Convert to CQP: `[word="fo."]`
3. Custom wildcard matching logic

**Issues**:
- Metacharacter escaping conflicts between simple and regex syntax
- Performance implications of complex regex patterns
- Handling of unicode characters

### Gap Token Semantics

**Simple Query**: `"fox + over"` (required gap)
**Problem**: Gap tokens have different semantics than CQP quantifiers

**CQP Equivalent**: `[word="fox"] []+ [word="over"]`

**Issues**:
- Multiple consecutive gap tokens: `"fox +++** over"` (3-5 tokens)
- Interaction with proximity operators
- Performance of skip patterns

### Proximity Operators

**Simple Query**: `"day <<3>> night"` (within 3 tokens)
**Problem**: No direct CQP equivalent for bidirectional proximity

**Potential CQP Translation**:
```
([word="day"] []{0,3} [word="night"]) | ([word="night"] []{0,3} [word="day"])
```

**Issues**:
- Exponential explosion with multiple proximity constraints
- Directional proximity operators (`<<3<<`, `>>3>>`)
- Same-sentence proximity (`<<s>>`) requires sentence boundary detection

### Alternative Lists

**Simple Query**: `"[car,truck,bus]"`
**CQP Equivalent**: `[word="car|truck|bus"]`

**Issues**:
- Wildcard patterns within alternatives: `"[ca*,tr*]"`
- Escaping commas and brackets
- Empty alternatives: `"neighbo[u,]r"`

### Regular Expression Groups

**Simple Query**: `"(very)? capable"`
**Problem**: Overlaps with but differs from CQP quantifiers

**Issues**:
- Nested groups and quantifiers
- Alternative groups: `"(red|blue) car"`
- Complex patterns: `"((very)? big)+ house"`

## Implementation Phases

### Phase 1: Core Parser Infrastructure

**Goals**:
- Basic pyparsing grammar for simple queries
- AST node classes for different query types
- Error handling and validation

**Deliverables**:
- `simple_parser.py` with pyparsing grammar
- AST classes for query components
- Basic test coverage

**Challenges**:
- Handling metacharacter escaping correctly
- Precedence rules for different operators
- Good error messages for invalid syntax

### Phase 2: Basic Feature Implementation

**Goals**:
- Simple word searches
- Basic wildcards (`?`, `*`, `+`)
- Word sequences
- Alternative lists

**Deliverables**:
- Compilation functions for basic features
- Integration with existing matcher
- Comprehensive test suite

**Challenges**:
- Regex pattern generation and escaping
- Case-insensitive matching by default
- Performance optimization

### Phase 3: Advanced Features

**Goals**:
- Gap tokens and complex quantifiers
- Regular expression groups
- Proximity operators (basic)

**Deliverables**:
- Full feature parity with grammar specification
- Performance benchmarks
- Documentation

**Challenges**:
- Complex opcode generation for gap patterns
- Proximity operator implementation
- Memory usage optimization

### Phase 4: Optimization and Integration

**Goals**:
- Performance optimization
- Better error messages
- Integration with existing search interface

**Deliverables**:
- Optimized query compilation
- User-friendly error reporting
- Documentation and examples

**Challenges**:
- Query optimization heuristics
- Error message localization
- Backward compatibility

## Technical Decisions Needed

### Default Column Name

**Question**: What column should simple queries search by default?

**Options**:
1. `"token"` (most common)
2. `"word"` (matches CQP convention)
3. Configurable default
4. Require explicit column specification

### Case Sensitivity

**Question**: How to handle case sensitivity?

**Options**:
1. Case-insensitive by default (matches BNCweb)
2. Case-sensitive by default (matches CQP)
3. Configurable default
4. Explicit modifiers only

### Proximity Implementation

**Question**: How to implement proximity operators efficiently?

**Options**:
1. Translate to complex CQP disjunctions
2. Custom proximity opcodes in Rust
3. Post-processing filter on results
4. Hybrid approach

### Error Handling

**Question**: How to provide good error messages?

**Options**:
1. Direct pyparsing error messages
2. Custom error message translation
3. Annotated error positions
4. Suggested corrections

## Testing Strategy

### Unit Tests

- Individual feature parsing
- AST node compilation
- Opcode generation
- Error handling

### Integration Tests

- End-to-end query execution
- Performance benchmarks
- Compatibility with CQP results
- Large corpus testing

### Property-Based Tests

- Query equivalence testing
- Regex pattern validation
- Performance bounds
- Memory usage limits

## Documentation Requirements

### User Documentation

- Query language syntax guide
- Examples and tutorials
- Migration guide from CQP
- Performance tips

### Developer Documentation

- Parser architecture overview
- AST node reference
- Compilation process
- Extension guidelines

## Performance Considerations

### Query Compilation Time

- Caching compiled queries
- Optimizing regex generation
- Minimizing opcode sequences

### Execution Performance

- Efficient wildcard matching
- Proximity operator optimization
- Memory usage optimization
- Parallel processing support

### Scalability

- Large corpus support
- Complex query handling
- Memory management
- Error recovery

## Risk Assessment

### High Risk

- Proximity operator complexity
- Performance regression vs CQP
- Feature interaction bugs
- Memory usage explosion

### Medium Risk

- Regex pattern generation errors
- Error message quality
- Testing coverage gaps
- Documentation completeness

### Low Risk

- Basic wildcard implementation
- Alternative list parsing
- Integration with existing matcher
- Simple word searches

## Success Criteria

### Functional

- All grammar features implemented
- Feature parity with specification
- Comprehensive test coverage
- Good error messages

### Performance

- No significant regression vs CQP
- Reasonable compilation times
- Efficient memory usage
- Scalable to large corpora

### Usability

- Intuitive syntax
- Clear documentation
- Easy migration from CQP
- Good developer experience