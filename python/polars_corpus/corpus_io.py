"""Read corpora of tagged text files into Polars frames."""

from __future__ import annotations

import os
from itertools import islice
from os import PathLike
from typing import Generator, Iterator, Optional, Union

import polars as pl
from polars.io.plugins import register_io_source

from ._internal import WlpFileReader

__all__ = ["read_text_corpus", "scan_text_corpus", "read_wlp_corpus", "scan_wlp_corpus"]


PathType = Union[str, bytes, PathLike[str], PathLike[bytes]]

# Rows per frame when the query engine asks for no particular batch size.
BATCH_SIZE = 10000


class CorpusReader:
    """Read a list of corpus files into a frame, one row per token.

    A subclass supplies either `read_file`, which turns one file into rows, or
    `read_frames`, when the format is parsed in Rust. This class adds the file
    id to each row and holds the eager and lazy entry points, none of which
    depend on the file format.

    The paths are listed at construction, so a generator can be passed and the
    files still re-read on every collect.
    """

    def __init__(self, corpus_files: Iterator[PathType]):
        self._corpus_files = list(corpus_files)

    def schema(self) -> pl.Schema:
        """The columns a file of this format yields, in order."""
        raise NotImplementedError()

    def read_file(self, path: PathType) -> Generator[dict[str, str]]:
        """Yield one row per token in `path`, in `schema` order less `file_id`,
        which `read_files` fills in.
        """
        raise NotImplementedError()

    def read_files(self) -> Generator[dict[str, str]]:
        """Yield the rows of every corpus file, each carrying its file id."""
        for file in self._corpus_files:
            # The full path rather than the basename, so two corpus files of
            # the same name in different directories stay distinct.
            file_id = os.fsdecode(file)
            for row in self.read_file(file):
                row["file_id"] = file_id
                yield row

    def read_frames(self, batch_size: int = BATCH_SIZE) -> Generator[pl.DataFrame]:
        """Yield the corpus as frames of at most `batch_size` rows.

        The default builds them from `read_file`'s rows; a format parsed in
        Rust overrides this and never goes through rows at all. Either way a
        batch is parsed only once it is asked for.
        """
        rows = self.read_files()
        while batch := list(islice(rows, batch_size)):
            yield pl.from_records(batch, schema=self.schema(), orient="row")

    def read_corpus(self) -> pl.DataFrame:
        """Read every corpus file into one DataFrame.

        The declared schema leads the concat, so a corpus holding no tokens
        still comes back with its columns.
        """
        frames = [pl.DataFrame(schema=self.schema()), *self.read_frames()]
        return pl.concat(frames, rechunk=True)

    def scan_corpus(self) -> pl.LazyFrame:
        """Register the reader as a Polars IO source, and hand back a LazyFrame.

        The schema is declared here rather than read off the rows, because the
        engine needs it before any file is opened. Batches are produced as the
        engine asks for them. The columns it selects and the filter it pushes
        down are applied to each batch, so a query over a corpus too large to
        hold never holds it.

        Based on https://docs.pola.rs/user-guide/plugins/io_plugins/#writing-the-source
        """

        def source_generator(
            with_columns: Optional[list[str]],
            predicate: Optional[pl.Expr],
            n_rows: Optional[int],
            batch_size: Optional[int],
        ) -> Iterator[pl.DataFrame]:
            batch_size = batch_size or BATCH_SIZE
            # A row limit caps the batch too, so a `head` of a huge corpus
            # parses the rows it wants and stops rather than a whole batch.
            if n_rows is not None:
                batch_size = min(batch_size, n_rows)

            for df in self.read_frames(batch_size):
                if n_rows is not None:
                    n_rows -= df.height

                if with_columns is not None:
                    df = df.select(with_columns)

                if predicate is not None:
                    df = df.filter(predicate)

                yield df

                if n_rows == 0:
                    break

        return register_io_source(io_source=source_generator, schema=self.schema())


class TextCorpusReader(CorpusReader):
    """Read files of `word/TAG` tokens, one sentence to a line.

    The split is at the last "/", so a token like `and/or/CC` comes out as the
    word `and/or` with the tag `CC`. Blank lines are skipped, and a token with
    no "/" in it is an error rather than a dropped row.
    """

    def schema(self) -> pl.Schema:
        return pl.Schema(
            {
                "token": pl.String,
                "pos": pl.String,
                "sentence_tag": pl.String,
                "file_id": pl.String,
            }
        )

    def read_file(self, path: PathType) -> Generator[dict[str, str]]:
        bos = True
        for line in open(path, "rt"):
            if line != "\n":
                bos = True
                tokens = line.strip().split()
                for token in tokens:
                    try:
                        tok, pos = token.rsplit("/", 1)
                    except ValueError:
                        raise ValueError(f'Malformed token "{token}"')
                    yield {
                        "token": tok,
                        "pos": pos,
                        "sentence_tag": "B" if bos else "I",
                    }
                    bos = False


def read_text_corpus(corpus_files: Iterator[PathType]) -> pl.DataFrame:
    """
    Read tagged text files into a DataFrame, one row per token.

    Each line of a file is one sentence, and each whitespace-separated token on
    it is a word and its tag joined by "/", e.g., `quick/JJ`. The split is at
    the last "/", so `and/or/CC` reads as the word `and/or` with the tag `CC`.
    Blank lines are skipped.

    Parameters
    ----------
    corpus_files : iterable of str or Path
        Paths of the files to read, e.g. `Path("corpus").glob("*.txt")`.

    Returns
    -------
    DataFrame
        One row per token, with columns `token`, `pos`, `sentence_tag` and
        `file_id`. The files come in the order given, and the tokens of a file
        in the order they appear in it. `sentence_tag` is "B" on the first
        token of a sentence and "I" on the rest. `file_id` holds the path the
        token was read from.

    Raises
    ------
    FileNotFoundError
        If one of `corpus_files` does not exist.
    ValueError
        If a token has no "/" in it.

    Notes
    -----
    Files are assumed to be encoded in UTF-8.

    See Also
    --------
    scan_text_corpus : Read the same files a batch at a time, as a LazyFrame.
    chunk_id : Number the sentences that `sentence_tag` marks out.

    Examples
    --------
    >>> from pathlib import Path
    >>> import polars_corpus as plc
    >>> corpus = plc.read_text_corpus(Path("corpus").glob("*.txt"))
    >>> # Sort the paths, so the file ids run in a predictable order:
    >>> corpus = plc.read_text_corpus(sorted(Path("corpus").glob("*.txt")))
    >>> plc.search(corpus, "the _JJ _NN")
    """
    return TextCorpusReader(corpus_files).read_corpus()


def scan_text_corpus(corpus_files: Iterator[PathType]) -> pl.LazyFrame:
    """
    Scan tagged text files as a LazyFrame, a batch of tokens at a time.

    Each line of a file is one sentence, and each whitespace-separated token on
    it is a word and its tag joined by "/", e.g., `quick/JJ`. The split is at
    the last "/", so `and/or/CC` reads as the word `and/or` with the tag `CC`.
    Blank lines are skipped.

    Parameters
    ----------
    corpus_files : iterable of str or Path
        Paths of the files to read, e.g. `Path("corpus").glob("*.txt")`.

    Returns
    -------
    LazyFrame
        One row per token, with columns `token`, `pos`, `sentence_tag` and
        `file_id`. The files come in the order given, and the tokens of a file
        in the order they appear in it. `sentence_tag` is "B" on the first
        token of a sentence and "I" on the rest. `file_id` holds the path the
        token was read from.

    Raises
    ------
    polars.exceptions.ComputeError
        On collecting, if one of `corpus_files` does not exist or a token has
        no "/" in it. The `FileNotFoundError` or `ValueError` behind it is
        named in the message.

    Notes
    -----
    Files are assumed to be encoded in UTF-8.

    See Also
    --------
    read_text_corpus : Read the same files into a DataFrame.

    Examples
    --------
    >>> from pathlib import Path
    >>> import polars as pl
    >>> import polars_corpus as plc
    >>> corpus = plc.scan_text_corpus(sorted(Path("corpus").glob("*.txt")))
    >>> corpus.select(pl.len()).collect()  # Count tokens without holding them
    >>> # search() reads a lazy corpus a chunk of files at a time:
    >>> plc.search(corpus, "the _JJ _NN")
    """
    return TextCorpusReader(corpus_files).scan_corpus()


class WlpCorpusReader(CorpusReader):
    """Read COCA/COHA `.wlp` files: a "##" line naming a text, then one token
    to a line as word, lemma and tag, tab-separated.

    The texts a file holds name themselves, so `file_id` comes from the "##"
    lines rather than the path. Parsing is done in Rust, a batch of tokens at
    a time, so `read_file` is unused.
    """

    def schema(self) -> pl.Schema:
        return pl.Schema(
            {
                "token": pl.String,
                "lemma": pl.String,
                "pos": pl.String,
                "file_id": pl.String,
            }
        )

    def read_frames(self, batch_size: int = BATCH_SIZE) -> Generator[pl.DataFrame]:
        for path in self._corpus_files:
            yield from WlpFileReader(os.fsdecode(path), batch_size)


def read_wlp_corpus(corpus_files: Iterator[PathType]) -> pl.DataFrame:
    return WlpCorpusReader(corpus_files).read_corpus()


def scan_wlp_corpus(corpus_files: Iterator[PathType]) -> pl.LazyFrame:
    return WlpCorpusReader(corpus_files).scan_corpus()
