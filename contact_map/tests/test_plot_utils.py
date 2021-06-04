
# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *

from contact_map.plot_utils import *
from contact_map.plot_utils import _ContactPlotRange
try:
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = True

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

@pytest.mark.parametrize("cmap", ['seismic', 'Blues', 'custom'])
def test_is_cmap_diverging(cmap):
    custom = None
    if HAS_MATPLOTLIB:
        custom = LinearSegmentedColormap(
            'testCmap',
            segmentdata={
                'red':   [[0.0,  0.0, 0.0],
                          [0.5,  1.0, 1.0],
                          [1.0,  1.0, 1.0]],
                 'green': [[0.0,  0.0, 0.0],
                           [0.25, 0.0, 0.0],
                           [0.75, 1.0, 1.0],
                           [1.0,  1.0, 1.0]],
                 'blue':  [[0.0,  0.0, 0.0],
                           [0.5,  0.0, 0.0],
                           [1.0,  1.0, 1.0]]},
            N=256
        )
    elif cmap == 'custom':
        # no matplotlib; can't do custom
        pytest.skip()

    cmap, expected = {
        'seismic': ('seismic', True),
        'Blues': ('Blues', False),
        'custom': (custom, False),
    }[cmap]
    if cmap == 'custom':
        with pytest.warns(UserWarning):
            assert is_cmap_diverging(cmap) == expected
    else:
        assert is_cmap_diverging(cmap) == expected


class TestContactRange(object):
    def setup(self):
        self.cr = _ContactPlotRange(5)
    
    @pytest.mark.parametrize("case", [(_ContactPlotRange(5), True),
                                      (_ContactPlotRange(3), False),
                                      (5, True),
                                      ((0, 5), False),  # Maybe support this?
                                      ({5}, False)  # Fail for non-int/tuple
                                      ])
                                      def test_eq(self, case):
                                          o, result = case
                                              assert (self.cr == o) is result
                                                  assert (self.cr != o) is not result
s
