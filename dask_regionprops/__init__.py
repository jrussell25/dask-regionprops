try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
__author__ = "John Russell"
__email__ = "johncrussell25@gmail.com"

from .regionprops import regionprops, regionprops_df

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "regionprops",
    "regionprops_df",
]
