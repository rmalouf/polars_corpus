from __future__ import annotations

import importlib

from .assoc import *  # noqa: F403
from .chunk import *  # noqa: F403
from .convert import *  # noqa: F403
from .cqp_parser import *  # noqa: F403
from .exprs import *  # noqa: F403
from .io import *  # noqa: F403
from .lexical import *  # noqa: F403
from .matcher import *  # noqa: F403

# from .productivity import *  # noqa: F403
from .search import *  # noqa: F403
from .utils import *  # noqa: F403
from .view import *  # noqa: F403
from .visualizations import *  # noqa: F403

__version__ = importlib.metadata.version("polars-corpus")
