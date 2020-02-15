import numpy as np

try:  # try loop for testing
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
except ImportError:  # pragma: no cover
    pass

def _get_cmap(cmap):
    """Get the appropriate matplotlib colormap.

    If the input is a string, it finds the associated matplotlib ColorMap
    object. If input is already a ColorMap, it is passed through.
    """
    if isinstance(cmap, str):
        cmap_f = plt.get_cmap(cmap)
    else:
        cmap_f = cmap

    return cmap_f


def _make_norm(rescale_range='auto'):
    """Create the matplotlib Normalize object
    """
    if rescale_range is None:
        norm = Normalize(vmin=0, vmax=1, clip=True)
    elif rescale_range == 'auto':
        norm = Normalize()
    elif isinstance(rescale_range, Normalize):
        norm = rescale_range
    else:
        vmin, vmax = rescale_range
        norm = Normalize(vmin, vmax, clip=True)

    return norm


def rgb_colors(data, cmap, rescale_range='auto', bytes=False):
    """RGB values representing colors after applying a colormap to data.

    Parameters
    ----------
    data : array-like
        Input data, should be 1-dimensional
    cmap : str or matplotlib.colors.Colormap
        Colormap to use
    rescale_range : matplotlib.colors.Normalize or str or None or tuple
        Defines the color normalization. If a ``Normalize`` object, that is
        passed through. The string 'auto' is interpresent as
        ``Normalize()``. No other strings are currently allowed. ``None`` is
        interpreted as ``Normalize(0, 1)``, meaning that the data is already
        mapped to the correct values. A tuple should be a 2-tuple of (vmin,
        vmax).
    bytes : bool
    """
    cmap_f = _get_cmap(cmap)
    norm = _make_norm(rescale_range)
    rescaled = norm(data)
    colors = cmap_f(rescaled, bytes=bytes)
    return colors


def int_colors(data, cmap, rescale_range='auto'):
    """Integers representing colors after applying a colormap to data.

    Parameters
    ----------
    data : array-like
        Input data, should be 1-dimensional
    cmap : str or matplotlib.colors.Colormap
        Colormap to use
    rescale_range : matplotlib.colors.Normalize or str or None or tuple
        Defines the color normalization. If a ``Normalize`` object, that is
        passed through. The string 'auto' is interpresent as
        ``Normalize()``. No other strings are currently allowed. ``None`` is
        interpreted as ``Normalize(0, 1)``, meaning that the data is already
        mapped to the correct values. A tuple should be a 2-tuple of (vmin,
        vmax).

    Returns
    -------
    list of int :
        single integer representation of each color
    """
    rgb = rgb_colors(data, cmap, rescale_range, bytes=True)
    colors = [256*256*c[0] + 256*c[1] + c[2] for c in rgb]
    return colors


def hex_colors(data, cmap, rescale_range='auto', style='web'):
    """Hex string representing colors after applying a colormap to data.

    Parameters
    ----------
    data : array-like
        Input data, should be 1-dimensional
    cmap : str or matplotlib.colors.Colormap
        Colormap to use
    rescale_range : matplotlib.colors.Normalize or str or None or tuple
        Defines the color normalization. If a ``Normalize`` object, that is
        passed through. The string 'auto' is interpresent as
        ``Normalize()``. No other strings are currently allowed. ``None`` is
        interpreted as ``Normalize(0, 1)``, meaning that the data is already
        mapped to the correct values. A tuple should be a 2-tuple of (vmin,
        vmax).
    style : str
        Which string representation to report. Valid values, with their
        examples of how they would represent the color red:
        * ``'web'``: ``#ff0000``
        * ``'python'``: ``0xff0000``
        * ``'raw'``: ``ff0000``

    Returns
    -------
    list of str :
        hexadecimal representation of each color
    """
    prefix = {'web': '#', 'python': '0x', 'raw': ''}[style]
    formatter = prefix + "{:06x}"
    ints = int_colors(data, cmap, rescale_range)
    return [formatter.format(c) for c in ints]


def ranged_colorbar(cmap, norm, cbmin, cbmax, name="Partial Map"):
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
    name : str
        name for the submap to be created

    Returns
    -------
    matplotlib.colorbar.Colorbar
        a colorbar restricted to the range given by cbmin, cbmax
    """
    # see https://stackoverflow.com/questions/24746231
    cmap_f = _get_cmap(cmap)
    cbmin_normed = float(cbmin - norm.vmin) / (norm.vmax - norm.vmin)
    cbmax_normed = float(cbmax - norm.vmin) / (norm.vmax - norm.vmin)
    n_colors = int(round((cbmax_normed - cbmin_normed) * cmap_f.N))
    colors = cmap_f(np.linspace(cbmin_normed, cbmax_normed, n_colors))
    new_cmap = LinearSegmentedColormap.from_list(name=name, colors=colors)
    new_norm = matplotlib.colors.Normalize(vmin=cbmin, vmax=cbmax)
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=new_norm)
    sm._A = []
    cb = plt.colorbar(sm, fraction=0.046, pad=0.04)
    return cb
