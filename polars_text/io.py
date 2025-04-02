from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Generator, Iterator, Union

import polars as pl
from polars.io.plugins import register_io_source

__all__ = ["read_text_corpus", "scan_text_corpus"]


PathType = Union[str, bytes, PathLike[str], PathLike[bytes]]


class CorpusReader:
    def __init__(self, corpus_files: Iterator[PathType]):
        self._corpus_files = list(corpus_files)

    def read_file(self, path: PathType) -> Generator[dict[str, str], None, None]:
        raise NotImplementedError()

    def read_files(self) -> Generator[dict[str, str], None, None]:
        for file in self._corpus_files:
            yield from self.read_file(file)

    def read_corpus(self) -> pl.DataFrame:
        return pl.DataFrame(self.read_files())

    def scan_corpus(self) -> pl.LazyFrame:
        """Based on https://docs.pola.rs/user-guide/plugins/io_plugins/#writing-the-source"""
        schema = pl.Schema({"token": pl.String, "tag": pl.String, "sent": pl.String})

        def source_generator(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            if batch_size is None:
                batch_size = 10000
            # Initialize the reader.
            reader = iter(self.read_files())
            # Ensure we don't read more rows than requested from the engine
            while n_rows is None or n_rows > 0:
                if n_rows is not None:
                    batch_size = min(batch_size, n_rows)

                rows = []

                for _ in range(batch_size):
                    try:
                        row = next(reader)
                    except StopIteration:
                        n_rows = 0
                        break
                    rows.append(row)

                df = pl.from_records(rows, schema=schema, orient="row")
                if n_rows is not None:
                    n_rows -= df.height

                # If we would make a performant reader, we would not read these
                # columns at all.
                if with_columns is not None:
                    df = df.select(with_columns)

                # If the source supports predicate pushdown, the expression can be parsed
                # to skip rows/groups.
                if predicate is not None:
                    df = df.filter(predicate)

                yield df

        return register_io_source(io_source=source_generator, schema=schema)


class TextCorpusReader(CorpusReader):
    def read_file(self, file: PathType) -> Generator[dict[str, str], None, None]:
        bos = True
        for line in open(file, "rt"):
            if line != "\n":
                bos = True
                tokens = line.strip().split()
                for token in tokens:
                    try:
                        tok, tag = token.rsplit("/", 1)
                    except ValueError:
                        raise ValueError(f'Malformed token "{token}"')
                    yield {"token": tok, "tag": tag, "sent": "B" if bos else "I"}
                    bos = False


def read_text_corpus(corpus_files: Iterator[PathType]) -> pl.DataFrame:
    return TextCorpusReader(corpus_files).read_corpus()


def scan_text_corpus(corpus_files: Iterator[PathType]) -> pl.LazyFrame:
    return TextCorpusReader(corpus_files).scan_corpus()
