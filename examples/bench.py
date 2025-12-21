import polars as pl
import polars_corpus as plc
import time

c = pl.read_parquet("examples/bnc.parquet")
now = time.time()
#result = c.corpus.search("( _{ADJ} )+ small ( _{ADJ} )* _{SUBST}", pos_column="tag")
result = c.corpus.search_cqp('$x:([pos!="ADJ"]) $y:([token="small"]) $z:([pos="ADJ"]*) [pos="SUBST"]')
print(f'{time.time()-now:.3f}', result)

# now = time.time()
# result = c.corpus.search_cqp('([pos!="ADJ"]) ([token="small"]) ([pos="ADJ"]*) [pos="SUBST"]')
#
# print(f'{time.time()-now:.3f}', result)

