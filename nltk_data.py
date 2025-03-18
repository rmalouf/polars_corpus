from collections import defaultdict
import nltk
import polars as pl
from nltk.tag.mapping import map_tag

nltk.download('brown')
nltk.download('universal_tagset')

## Brown corpus

data = defaultdict(list)
for cat in nltk.corpus.brown.categories():
    for fileid in nltk.corpus.brown.fileids(categories=cat):
        for i, sent in enumerate(nltk.corpus.brown.tagged_sents(fileid)):
            first = True
            for tok, tag in sent:
                data['tok'].append(tok)
                data['tag'].append(tag)
                data['pos'].append(map_tag('brown','universal',tag))
                data['fileid'].append(fileid)
                if first:
                    data['sent'].append('B')
                    first = False
                else:
                    data['sent'].append('I')
                data['cat'].append(cat)

c = pl.DataFrame(data, schema={'tok':pl.String,
                               'tag':pl.String,
                               'pos':pl.String,
                               'fileid':pl.String,
                               'sent':pl.Categorical,
                               'cat':pl.Categorical})

c.write_parquet('brown.parquet')
