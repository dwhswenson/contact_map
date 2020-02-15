
# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *

from contact_map.plot_utils import *
from contact_map.plot_utils import _get_cmap, _make_norm

@pytest.mark.parametrize("cmap_", ["seismic", "cmap"])
def test_get_cmap(cmap_):
    matplotlib = pytest.importorskip("matplotlib")
    cmapf_f_real = matplotlib.pyplot.get_cmap('seismic')
    if cmap_ == "cmap":
        cmap_ = cmapf_f_real


    cmap = _get_cmap(cmap_)
    assert cmap is cmapf_f_real

@pytest.mark.parametrize("rescale_range", [None, 'auto', 'tuple', 'mpl'])
def test_make_norm(rescale_range):
    matplotlib = pytest.importorskip("matplotlib")
    vmin, vmax, clip = {None: (0, 1, True),
                        'auto': (None, None, False),
                        'tuple': (-1, 1, True),
                        'mpl': (-1, 1, False)}[rescale_range]
    expected = matplotlib.colors.Normalize(vmin, vmax, clip)
    if rescale_range == 'tuple':
        rescale_range = vmin, vmax
    elif rescale_range == 'mpl':
        rescale_range = expected
    norm = _make_norm(rescale_range)
    assert norm.vmin == expected.vmin
    assert norm.vmax == expected.vmax
    assert norm.clip == expected.clip
    if rescale_range == 'mpl':
        assert norm is expected

@pytest.mark.parametrize('result_type', ['bytes', 'float'])
def test_rgb_colors(result_type):
    matplotlib = pytest.importorskip("matplotlib")
    values = [0.0, 0.5, 1.0]
    use_bytes = {'bytes': True, 'float': False}[result_type]
    expected = [(0, 0, 255, 255),
                (255, 254, 254, 255),  # bwr gives *almost* white at 0.5
                (255, 0, 0, 255)]
    if not use_bytes:
        expected = (np.array(expected) / 255.0).tolist()

    colors = rgb_colors(values, cmap='bwr', bytes=use_bytes)
    assert_allclose(colors, expected)

def test_int_colors():
    matplotlib = pytest.importorskip("matplotlib")
    values = [0.0, 0.5, 1.0]
    expected = [255, 16776958, 16711680]
    colors = int_colors(values, cmap='bwr')
    for c, e in zip(colors, expected):
        assert c == e

@pytest.mark.parametrize('style', ['raw', 'web', 'python'])
def test_hex_colors(style):
    matplotlib = pytest.importorskip("matplotlib")
    values = [0.0, 0.5, 1.0]
    expected = ['0000ff', 'fffefe', 'ff0000']
    prefix = {'raw': '', 'web': '#', 'python': '0x'}[style]
    colors = hex_colors(values, cmap='bwr', style=style)
    for c, e in zip(colors, expected):
        assert c == prefix + e


@pytest.mark.parametrize("val", [0.5, 0.55, 0.6, 0.65, 0.7])
@pytest.mark.parametrize("map_type", ["name", "cmap"])
def test_ranged_colorbar_cmap(map_type, val):
    matplotlib = pytest.importorskip("matplotlib")
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    default_cmap = matplotlib.pyplot.get_cmap('seismic')
    cmap = {
        'name': 'seismic',
        'cmap': default_cmap
    }[map_type]
    cbmin = 0.5
    cbmax = 0.7
    cb = ranged_colorbar(cmap, norm, cbmin, cbmax)
    # slight error from different discretizations
    atol = (norm.vmax - norm.vmin) / (cbmax - cbmin) / default_cmap.N
    assert np.allclose([atol], [0.0390625])  # to remind that this is small
    assert_allclose(cb.cmap(cb.norm(val)), default_cmap(norm(val)),
                    atol=atol)
