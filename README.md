specialist --targets nlpolars/cqp.py --output report try_cqp.py



t0 = time.time()
# c1 = bnc.select(pl.col('token', 'pos').corpus.kwic_concordance(r, 2))
# print(f"{time.time() - t0:.3f}")
# del c1
# 
# t0 = time.time()
# c2 = bnc.corpus.kwic_concordance(r, pl.col('token', 'pos'), 2)
# print(f"{time.time() - t0:.3f}")
# del c2
# 
# t0 = time.time()
# c3 = plc.kwic_concordance(r, pl.col('token', 'pos'), 2)
# print(f"{time.time() - t0:.3f}")
# del c3