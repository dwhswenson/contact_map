
# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *

from contact_map.plot_utils import *


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






