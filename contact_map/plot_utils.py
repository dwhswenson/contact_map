import numpy as np
import warnings
import collections

try:  # try loop for testing
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:  # pragma: no cover
    pass

_DIVERGING = [
    'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn',
    'Spectral', 'coolwarm', 'bwr', 'seismic',
]
_SEQUENTIAL = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples',
    'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd',
    'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn',
    'YlGn' 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
    'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot',
    'afmhot', 'gist_heat', 'copper'
]


def is_cmap_diverging(cmap):
    if cmap in _DIVERGING:
        return True
    elif cmap in _SEQUENTIAL:
        return False
    else:
        warnings.warn("Unknown colormap: Treating as sequential.")
        return False


def ranged_colorbar(cmap, norm, cbmin, cbmax, ax=None):
    """Create a colorbar with given endpoints.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        the base colormap to use
    norm : matplotlib.colors.Normalize
        the normalization (range of values) used in the image
    cbmin : float
        minimum value for the colorbar
    cbmax : float
        maximum value for the colorbar
    ax : matplotlib.Axes
        the axes to take space from to plot the colorbar

    Returns
    -------
    matplotlib.colorbar.Colorbar
        a colorbar restricted to the range given by cbmin, cbmax
    """
    # see https://stackoverflow.com/questions/24746231
    if isinstance(cmap, str):
        cmap_f = plt.get_cmap(cmap)
    else:
        cmap_f = cmap

    if ax is None:
        fig = plt
        ax = plt.gca()
    else:
        fig = ax.figure

    cbmin_normed = float(cbmin - norm.vmin) / (norm.vmax - norm.vmin)
    cbmax_normed = float(cbmax - norm.vmin) / (norm.vmax - norm.vmin)
    n_colors = int(round((cbmax_normed - cbmin_normed) * cmap_f.N))
    colors = cmap_f(np.linspace(cbmin_normed, cbmax_normed, n_colors))
    new_cmap = LinearSegmentedColormap.from_list(name="Partial Map",
                                                 colors=colors)
    new_norm = matplotlib.colors.Normalize(vmin=cbmin, vmax=cbmax)
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=new_norm)
    sm._A = []
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    return cb


def _int_or_range_to_tuple(posible_int):
    if isinstance(posible_int, collections.abc.Iterable):
        return (posible_int[0], posible_int[1])
    else:
        return (0, posible_int)


def _get_low_high_counter_range(counter):
    """Give the (min, max + 1) for both the low and high keys in counter"""
    keys = [tuple(sorted(list(i))) for i in counter.keys()]
    if len(keys) == 0:
        return (0, 0), (0, 0)
    lows, highs = zip(*keys)
    return (min(lows), max(lows)+1), (min(highs), max(highs)+1)


def _get_sorted_counter_range(counter):
    """Return smallest range, longest range for the low and high counter"""
    low, high = _get_low_high_counter_range(counter)
    if low[1]-low[0] > high[-1]-high[0]:
        return high, low
    else:
        return low, high


def _sanitize_n_x_n_y(n_x, n_y, counter):
    if n_x is None and n_y is None:
        n_x, n_y = _get_sorted_counter_range(counter)
    elif n_x is None or n_y is None:
        raise ValueError("Either both n_x and n_y need to be defined or "
                         "neither.")
    if isinstance(n_x, _ContactPlotRange):
        n_x = n_x.n
    if isinstance(n_y, _ContactPlotRange):
        n_y = n_y.n
    return n_x, n_y


def make_x_y_ranges(n_x, n_y, counter):
    """Return ContactPlotRange for both x and y"""
    n_x, n_y = _sanitize_n_x_n_y(n_x, n_y, counter)
    n_x = _ContactPlotRange(n_x)
    n_y = _ContactPlotRange(n_y)
    return n_x, n_y


class _ContactPlotRange(object):
    """Object that deals with functions that are needed for plot ranges

    Parameters
    ----------
    n : int, tuple(start, end)
        range of objects in the given direction (used in plotting)
    """
    def __init__(self, n):
        self.n = n
        self.min, self.max = _int_or_range_to_tuple(n)

    @property
    def range_length(self):
        return self.max - self.min

    def __eq__(self, other):
        if isinstance(other, (int, tuple)):
            return self.n == other
        elif isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
