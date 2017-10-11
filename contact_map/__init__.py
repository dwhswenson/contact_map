try:
    from . import version
except ImportError:
    from . import _version as version

__version__ = version.version

from .contact_map import (
    ContactMap, ContactFrequency, ContactDifference, ContactCount
)

from .min_dist import NearestAtoms, MinimumDistanceCounter

# import concurrence
