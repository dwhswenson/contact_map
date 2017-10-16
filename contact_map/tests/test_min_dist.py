# pylint: disable=wildcard-import, unused-wildcard-import
# pylint: disable=wrong-import-order
# pylint: disable=missing-docstring, no-self-use, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name

import mdtraj as md

#includes pytest
from .utils import *

from contact_map.min_dist import *

traj = md.load(find_testfile("trajectory.pdb"))

@pytest.mark.parametrize("idx", [0, 4])
class TestNearestAtoms(object):
    def setup(self):
        self.nearest_atoms = {
            idx: NearestAtoms(traj, cutoff=0.075, frame_number=idx)
            for idx in [0, 4]
        }

    def test_initialization(self, idx):
        nearest = self.nearest_atoms[idx]
        expected_nearest_atoms = {
            0: {1: 4, 4: 6, 5: 6, 6: 5},
            4: {0: 9, 1: 8, 4: 6, 5: 7, 6: 4, 7: 5, 8: 1, 9: 0}
        }
        expected_nearest_distance = {
            0: {1: 0.070710678118654752440,  # 0.1*0.5*sqrt(2)
                4: 0.063639610306789277196,  # 0.1*sqrt(0.45^2 + 0.45^2)
                5: 0.045276925690687083132,  # 0.1*sqrt(0.05^2 + 0.45^2)
                6: 0.045276925690687083132},
            4: {0: 0.045,
                1: 0.045,
                4: 0.05,
                5: 0.05,
                6: 0.05,
                7: 0.05,
                8: 0.045,
                9: 0.045}
        }
        assert nearest.nearest == expected_nearest_atoms[idx]
        expected_distance = pytest.approx(expected_nearest_distance[idx])
        assert nearest.nearest_distance == expected_distance

    def test_sorted_distances(self, idx):
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
