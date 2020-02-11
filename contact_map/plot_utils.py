import numpy as np

try:  # try loop for testing
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:  # pragma: no cover
    pass

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
    if isinstance(cmap, str):
        cmap_f = plt.get_cmap(cmap)
    else:
        cmap_f = cmap
    cbmin_normed = float(cbmin - norm.vmin) / (norm.vmax - norm.vmin)
    cbmax_normed = float(cbmax - norm.vmin) / (norm.vmax - norm.vmin)
    n_colors = int(round((cbmax_normed - cbmin_normed) * cmap_f.N))
    colors = cmap_f(np.linspace(cbmin_normed, cbmax_normed, n_colors))
    new_cmap = LinearSegmentedColormap.from_list(name="Partial Map",
                                                 colors=colors)
    new_norm = matplotlib.colors.Normalize(vmin=cbmin, vmax=cbmax)
    sm = plt.cm.ScalarMappable(cmap=new_cmap, norm=new_norm)
    sm._A = []
    cb = plt.colorbar(sm, fraction=0.046, pad=0.04)
    return cb


