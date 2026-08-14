import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

file_ids = []
text_types = []
tokens = []
tags = []
lemmas = []
sentence_tags = []

files = sorted(list(Path("/Volumes/Corpora/amalgum/").glob("amalgum*/*/xml/*.xml")))
for f in files:
    # The files declare encoding="utf8", a name expat rejects.  Decoding here and
    # handing ElementTree a str makes it ignore the declaration.
    text = ET.fromstring(f.read_text(encoding="utf-8"))
    file_id = text.get("id")
    text_type = text.get("type")
    for s in text.findall(".//s"):
        first = True
        for chunk in s.itertext():
            for word in chunk.strip().split("\n"):
                if word:
                    token, tag, lemma = word.split("\t")
                    file_ids.append(file_id)
                    text_types.append(text_type)
                    tokens.append(token)
                    tags.append(tag)
                    lemmas.append(lemma)
                    sentence_tags.append("B" if first else "I")
                    first = False

df = pl.DataFrame(
    {
        "token": tokens,
        "pos": tags,
        "lemma": lemmas,
        "sentence_tag": sentence_tags,
        "file_id": file_ids,
        "text_type": text_types,
    }
)
df.write_parquet("amalgum.parquet")
