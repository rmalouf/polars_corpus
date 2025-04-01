from os import PathLike
from typing import Generator, Iterator, Union

import polars as pl
from polars.io.plugins import register_io_source

__all__ = ["read_corpus", "scan_corpus"]


PathType = Union[str, bytes, PathLike[str], PathLike[bytes]]


def read_file(file: PathType) -> Generator[dict[str, str], None, None]:
    bos = True
    for line in open(file, "rt"):
        if line != "\n":
            bos = True
            tokens = line.strip().split()
            for token in tokens:
                tok, tag = token.rsplit("/", 1)
                yield {"token": tok, "tag": tag, "sent": "B" if bos else "I"}
                bos = False


def read_files(corpus_files: list[PathType]) -> Generator[dict[str, str], None, None]:
    for file in corpus_files:
        yield from read_file(file)


def read_corpus(corpus_files: list[PathType]) -> pl.DataFrame:
    return pl.DataFrame(read_files(corpus_files))


def scan_corpus(corpus_files: list[PathType]) -> pl.LazyFrame:
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
        reader = iter(read_files(corpus_files))
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
