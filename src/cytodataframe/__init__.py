"""
Initialization for cytodataframe package
"""

from . import engine
from .frame import CytoDataFrame
from .lazy import CytoLazyFrame
from .schema import CytoSchema

# Resolve the installed package version. Prefer the setuptools-scm generated
# ``_version.py`` (written during build / editable install and kept in step with
# the repo's git state); fall back to the installed package metadata, and
# finally to a placeholder when running from an unbuilt source tree.
try:
    from ._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("cytodataframe")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "CytoDataFrame",
    "CytoLazyFrame",
    "CytoSchema",
    "engine",
]
