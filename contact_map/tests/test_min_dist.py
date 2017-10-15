import mdtraj as md

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

#includes pytest
from .utils import *

from contact_map.min_dist import *

traj = md.load(find_testfile("trajectory.pdb"))

class TestNearestAtoms(object):
    def setup(self):
        pass

    def test_initialization(self):
        pytest.skip()

    def test_sorted_distances(self):
        pytest.skip()

class TestMinimumDistanceCounter(object):
    def setup(self):
        pass

    def test_initialization(self):
        pytest.skip()

    def test_from_contact_map(self):
        pytest.skip()

    def test_remap(self):
        pytest.skip()

    def test_atom_history(self):
        pytest.skip()

    def test_atom_count(self):
        pytest.skip()

    def test_residue_history(self):
        pytest.skip()

    def test_residue_count(self):
        pytest.skip()
