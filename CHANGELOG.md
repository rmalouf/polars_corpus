# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Polars 2.0 is now supported; the `<2.0` upper bound is gone.

### Fixed

- pyo3 is pinned to 0.29.2. Under 0.29.3 the extension segfaulted on import.

## [0.2.0] - 2026-09-29

Initial public release.

[Unreleased]: https://github.com/rmalouf/polars_corpus/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rmalouf/polars_corpus/releases/tag/v0.2.0
