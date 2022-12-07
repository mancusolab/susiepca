from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

# flat api; unfortunate but can work with autodoc
"""
from infer import *
from metrics import *
from sim import *
from utils import *
"""
from susiepca import infer, metrics, sim, utils

__all__ = ["infer", "metrics", "sim", "utils"]

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
