import pytest

cython = pytest.importorskip("cython")
from contact_map.contact_map import COMPILED


@pytest.mark.skipif(not COMPILED, reason="no compiled code")
def test_cythonization():
    assert COMPILED
