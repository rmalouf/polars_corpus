import polars as pl
import nlpolars as mp

df = pl.DataFrame({
    'a': [1, 1, None],
    'b': [4.1, 5.2, 6.3],
    'c': ['hello', 'everybody!', '!']
})
print(df.with_columns(mp.noop(pl.all()).name.suffix('_noop')))