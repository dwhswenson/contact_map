import pytest

cython = pytest.importorskip("cython")


def test_cythonization():
    from contact_map.contact_map import COMPILED
    assert COMPILED
