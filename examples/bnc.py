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
import pyarrow.parquet as pq


def get_xml(filename):
    doc = etree.parse(str(filename))
    docid = doc.xpath('//idno[@type="bnc"]')[0].text
    if text := doc.xpath("//wtext"):
        text_mode = "written"
        text_type = text[0].xpath("./@type")[0]
    elif text := doc.xpath("//stext"):
        text_mode = "spoken"
        text_type = text[0].xpath("./@type")[0]
    else:
        raise ValueError("Unknown text type")

    speakers = defaultdict(list)
    for person in doc.xpath("//person"):
        speakers["speaker_id"].append(
            person.get("{http://www.w3.org/XML/1998/namespace}id")
        )
        for attr in [
            "ageGroup",
            "soc",
            "sex",
            "persName",
            "occupation",
            "dialect",
            "persNote",
        ]:
            if (t := person.get(attr)) is None:
                speakers[attr].append(None)
            else:
                speakers[attr].append(t)

    speakers_df = pl.DataFrame(
        speakers,
        schema={
            "speaker_id": pl.String,
            "sex": pl.String,
            "ageGroup": pl.String,
            "dialect": pl.String,
            #'educ': pl.String,
            "soc": pl.String,
            "persName": pl.String,
            #'age': pl.String,
            "occupation": pl.String,
            "persNote": pl.String,
        },
    )

    data = defaultdict(list)
    for s in doc.xpath("//s"):
        if u := s.xpath("ancestor::u"):
            speaker_id = u[0].get("who")
        else:
            speaker_id = None
        sent_tag = "B"
        for token in s.xpath(
            ".//w | .//c | .//gap | .//unclear | .//pause | .//vocal | .//event"
        ):
            if (token.tag == "c" or token.tag == "w") and token.text is None:
                pass
            else:
                data["speaker_id"].append(speaker_id)
                data["mode"].append(text_mode)
                data["text_type"].append(text_type)
                data["file_id"].append(docid)
                data["sentence_tag"].append(sent_tag)
                if token.tag == "w":
                    data["token"].append(token.text.strip())
                    data["tag"].append(token.get("c5"))
                    data["lemma"].append(token.get("hw"))
                    data["pos"].append(token.get("pos"))
                elif token.tag == "c":
                    data["token"].append(token.text.strip())
                    data["tag"].append(token.get("c5"))
                    data["lemma"].append(None)
                    data["pos"].append("STOP")
                else:
                    data["token"].append(f"<{token.tag}/>")
                    data["tag"].append(None)
                    data["lemma"].append(None)
                    data["pos"].append(None)
            sent_tag = "I"

    corpus_df = pl.DataFrame(
        data,
        schema={
            "token": pl.String,
            "lemma": pl.String,
            "pos": pl.String,
            "tag": pl.String,
            "sentence_tag": pl.String,
            #            "mode": pl.Categorical,
            #            "text_type": pl.Categorical,
            #            "file_id": pl.Categorical,
            "mode": pl.String,
            "text_type": pl.String,
            "file_id": pl.String,
            "speaker_id": pl.String,
        },
    )
    return corpus_df, speakers_df


paths = sorted(list(Path("/Volumes/Corpora/bnc_xml/Texts").glob("**/*.xml")))
with joblib_progress(total=len(paths)):
    with Parallel(n_jobs=8, return_as="generator", verbose=0) as parallel:
        corpus_dfs, speakers_dfs = zip(
            *parallel(delayed(get_xml)(path) for path in paths)
        )
        corpus_df = pl.concat(corpus_dfs)
        speakers_df = pl.concat(speakers_dfs)

# corpus_df.write_parquet("bnc.parquet")

writer = None
for part in corpus_df.partition_by("file_id", maintain_order=True):
    table = part.to_arrow()
    if writer is None:
        writer = pq.ParquetWriter("bnc.parquet", table.schema)
    writer.write_table(table)  # each call = one row group

writer.close()


speakers_df.write_parquet("bnc-speakers.parquet")
