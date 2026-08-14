import polars as pl
from pathlib import Path
from lxml import etree
from tqdm.auto import tqdm

file_ids = []
text_types = []
tokens = []
tags = []
lemmas = []
sentence_tags = []

files = sorted(list(Path("/Volumes/Corpora/amalgum/").glob("amalgum*/*/xml/*.xml")))
for f in tqdm(files):
    tree = etree.parse(f)
    text = tree.getroot()
    file_id = text.get("id")
    text_type = text.get("type")
    for s in text.findall(".//s"):
        first = True
        chunks = s.xpath(".//text()")
        for chunk in chunks:
            for word in chunk.strip().split("\n"):
                if not word:
                    pass
                if word:
                    token, tag, lemma = word.split("\t")
                    file_ids.append(file_id)
                    text_types.append(text_type)
                    tokens.append(token)
                    tags.append(tag)
                    lemmas.append(lemma)
                    if first:
                        sentence_tags.append("B")
                        first = False
                    else:
                        sentence_tags.append("I")

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
