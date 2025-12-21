import polars as pl
from polars_corpus.matcher import get_matches

# Create test corpus
corpus = pl.DataFrame({
    "word": ["the", "big", "red", "house", "yesterday"],
    "pos": ["DT", "JJ", "JJ", "NN", "RB"],
})

# Test different quantifiers with bindings
test_cases = [
    ('$adjs: ([pos="JJ"]+) [pos="NN"]', "plus"),
    ('$adjs: ([pos="JJ"]*) [pos="NN"]', "star"),
    ('$det: ([pos="DT"]?) [pos="JJ"]', "optional"),
    ('$two: ([pos="JJ"]{2}) [pos="NN"]', "exact"),
]

for query, desc in test_cases:
    matches = get_matches(corpus, query)
    if matches:
        for i, m in enumerate(matches):
            print(f"{desc:10} match {i}: span={m.span}, bindings={m.bindings}")
    else:
        print(f"{desc:10} no matches")
