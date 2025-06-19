## Convert XML BNC into polars DataFrames and save a parquet files

# BNC XML to Parquet
#
# This notebook reads BNC XML files and converts them to a Parquet file using Polars.  The BNC XML files are available from the BNC website: https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554
# [Reference Guide for the British National Corpus (XML Edition)](http://www.natcorp.ox.ac.uk/docs/URG/index.html)

from collections import defaultdict
from pathlib import Path

import polars as pl
from lxml import etree

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

    speakers = defaultdict(list)
    for person in doc.xpath('//person'):
        speakers['speaker_id'].append(person.get('{http://www.w3.org/XML/1998/namespace}id'))
        for attr in ['ageGroup', 'soc', 'sex', 'persName', 'occupation', 'dialect', 'persNote']:
           if (t := person.get(attr)) is None:
               speakers[attr].append(None)
           else:
               speakers[attr].append(t)

    speakers_df = pl.DataFrame(speakers, schema={'speaker_id': pl.String,
                                           'sex': pl.String,
                                           'ageGroup': pl.String,
                                           'dialect': pl.String,
                                           #'educ': pl.String,
                                           'soc': pl.String,
                                           'persName': pl.String,
                                           #'age': pl.String,
                                           'occupation': pl.String,
                                           'dialect': pl.String,
                                           'persNote': pl.String
                                           })

    data = defaultdict(list)
    for s in doc.xpath('//s'):
        if u := s.xpath('ancestor::u'):
            speaker_id = u[0].get('who')
        else:
            speaker_id = None
        sent_tag = 'B'
        for token in s.xpath('.//w | .//c | .//gap | .//unclear | .//pause | .//vocal | .//event'):
            if (token.tag == 'c' or token.tag == 'w') and token.text is None:
                pass
            else:
                data['speaker_id'].append(speaker_id)
                data['mode'].append(text_mode)
                data['text_type'].append(text_type)
                data['file_id'].append(docid)
                data['sent_tag'].append(sent_tag)
                if token.tag == 'w':
                    data['token'].append(token.text.strip())
                    data['c5'].append(token.get('c5'))
                    data['lemma'].append(token.get('hw'))
                    data['pos'].append(token.get('pos'))
                elif token.tag == 'c':
                    data['token'].append(token.text.strip())
                    data['c5'].append(token.get('c5'))
                    data['lemma'].append(None)
                    data['pos'].append('STOP')
                else:
                    data['token'].append(f'<{token.tag}/>')
                    data['c5'].append(None)
                    data['lemma'].append(None)
                    data['pos'].append(None)
            sent_tag = 'I'

    c = pl.DataFrame(data, schema={'token': pl.String,
                                   'lemma': pl.String,
                                   'pos': pl.String,
                                   'c5': pl.String,
                                   'sent_tag': pl.String,
                                   'mode': pl.Categorical,
                                   'text_type': pl.Categorical,
                                   'file_id': pl.Categorical,
                                   'speaker_id': pl.String})
    return c, speakers_df

paths = sorted(list(Path('/Volumes/Corpora/bnc_xml').glob('**/*.xml')))
with joblib_progress(total=len(paths)):
    with Parallel(n_jobs=8, return_as="generator_unordered", verbose=0) as parallel:
        data1, data2 = zip(*parallel(delayed(get_xml)(path) for path in paths))
        c = pl.concat(data1)
        s = pl.concat(data2)

c.write_parquet('bnc.parquet')
s.write_parquet('bnc-speakers.parquet')
