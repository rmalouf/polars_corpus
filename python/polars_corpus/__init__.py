from __future__ import annotations

import importlib.metadata

from .assoc import *  # noqa: F403
from .chunk import *  # noqa: F403
from .convert import *  # noqa: F403
from .corpus_io import *  # noqa: F403
from .cqp_parser import *  # noqa: F403
from .dispersion import *  # noqa: F403
from .embeddings import *  # noqa: F403
from .exprs import *  # noqa: F403
from .keywords import *  # noqa: F403
from .lexical import *  # noqa: F403
from .matcher import *  # noqa: F403
from .search import *  # noqa: F403
from .utils import *  # noqa: F403
from .view import *  # noqa: F403

try:
    # matplotlib/seaborn are the "examples" extra, not a core dependency --
    # plotting is unavailable, but the rest of the package still works.
    from .visualizations import *  # noqa: F403
except ImportError:
    pass

__version__ = importlib.metadata.version("polars-corpus")
