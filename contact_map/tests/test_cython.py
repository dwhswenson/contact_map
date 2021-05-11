import pytest

cython = pytest.importorskip("cython")
COMPILED = pytest.importskip("contact_map.contact_map.COMPILED")


def test_cythonization():
    assert COMPILED
