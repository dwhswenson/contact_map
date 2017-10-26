try:
    from . import version
except ImportError:  # pragma: no cover
    from . import _version as version

__version__ = version.version

from .contact_map import (
    ContactMap, ContactFrequency, ContactDifference, ContactCount
)

from .min_dist import NearestAtoms, MinimumDistanceCounter

# import concurrence
