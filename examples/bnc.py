## Convert XML BNC into polars DataFrames and save a parquet files

# BNC XML to Parquet
#
# This notebook reads BNC XML files and converts them to a Parquet file using Polars.  The BNC XML files are available from the BNC website: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554
# [Reference Guide for the British National Corpus (XML Edition)](http://www.natcorp.ox.ac.uk/docs/URG/index.html)

from collections import defaultdict
from pathlib import Path

import polars as pl
from lxml import etree
from tqdm.auto import tqdm

from joblib import Parallel, delayed
from joblib_progress import joblib_progress

def get_xml(filename):
    doc = etree.parse(str(filename))
    docid = doc.xpath('//idno[@type="bnc"]')[0].text
    if text := doc.xpath('//wtext'):
        text_mode = 'written'
        text_type = text[0].xpath('./@type')[0]
    elif text := doc.xpath('//stext'):
        text_mode = 'spoken'
        text_type = text[0].xpath('./@type')[0]
    else:
        raise ValueError('Unknown text type')

    data = defaultdict(list)
    for s in doc.xpath('//s'):
        sent_tag = 'B'
        for item in s.xpath('.//w | .//c | .//gap | .//unclear | .//pause | .//vocal | .//event'):
            if (item.tag == 'c' or item.tag == 'w') and item.text is None:
                pass # print(docid, sentid, item.get('c5'))
            else:
                data['mode'].append(text_mode)
                data['text_type'].append(text_type)
                data['fileid'].append(docid)
                data['sent_tag'].append(sent_tag)
                if item.tag == 'w':
                    data['token'].append(item.text.strip())
                    data['c5'].append(item.get('c5'))
                    data['hw'].append(item.get('hw'))
                    data['pos'].append(item.get('pos'))
                elif item.tag == 'c':
                    data['token'].append(item.text.strip())
                    data['c5'].append(item.get('c5'))
                    data['hw'].append(None)
                    data['pos'].append('STOP')
                else:
                    data['token'].append(etree.tostring(item).decode())
                    data['c5'].append(None)
                    data['hw'].append(None)
                    data['pos'].append(None)
            sent_tag = 'I'

    c = pl.DataFrame(data, schema={'token': pl.String,
                                   'c5': pl.String,
                                   'pos': pl.String,
                                   'mode': pl.Categorical,
                                   'text_type': pl.Categorical,
                                   'fileid': pl.Categorical,
                                   'sent_tag': pl.String,
                                   'hw': pl.String})
    return c

paths = sorted(list(Path('/Volumes/Corpora/bnc_xml').glob('**/*.xml')))
with joblib_progress(total=len(paths)):
    with Parallel(n_jobs=8, return_as="generator_unordered", verbose=0) as parallel:
        data = parallel(delayed(get_xml)(path) for path in paths)
        c = pl.concat(data)

c.write_parquet('bnc.parquet')