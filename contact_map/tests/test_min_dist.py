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
        }[idx]
        expected_distance = {
            0: {1: 0.070710678118654752440,  # 0.1*0.5*sqrt(2)
                4: 0.063639610306789277196,  # 0.1*sqrt(0.45^2 + 0.45^2)
                5: 0.045276925690687083132,  # 0.1*sqrt(0.05^2 + 0.45^2)
                6: 0.045276925690687083132},
            4: {0: 0.045,
                1: 0.04501110973970759588,  # 0.1*sqrt(0.45^2 + 0.01^2)
                4: 0.051,
                5: 0.051,
                6: 0.051,
                7: 0.051,
                8: 0.04501110973970759588,
                9: 0.045}
        }[idx]
        assert nearest.nearest == expected_nearest_atoms
        assert nearest.nearest_distance == pytest.approx(expected_distance)

    def test_with_all_atoms(self, idx):
        n_atoms = traj.n_atoms
        distance_exceptions = {
            0: {5: 0.045276925690687083132,  # 0.1*sqrt(0.05^2 + 0.45^2)
                6: 0.045276925690687083132,
                7: 0.05522680508593630387},  # 0.1*sqrt(0.05^2 + 0.55^2)
            4: {0: 0.045,
                1: 0.04501110973970759588,  # 0.1*sqrt(0.45^2 + 0.01^2
                8: 0.04501110973970759588,  # 0.1*sqrt(0.45^2 + 0.01^2
                9: 0.045}
        }[idx]
        atom_exceptions = {
            0: {5: 6, 6: 5},
            4: {0: 9, 1: 8, 8: 1, 9: 0}
        }[idx]
        nearest = NearestAtoms(traj, cutoff=0.075, frame_number=idx,
                               excluded={})
        # in most cases we map pairs (dimers) to each other, with atom
        # numbers being grouped by value of i//2. That is, {0: 1, 1: 0,
        # 2: 3, 3: 2, ...}. This does that:
        expected_nearest_atoms = {i: i + (1 - 2 * (i % 2))
                                  for i in range(n_atoms)}
        expected_nearest_atoms.update(atom_exceptions)
        assert nearest.nearest == expected_nearest_atoms

        expected_distance = {i: 0.05 for i in range(n_atoms)}
        expected_distance.update(distance_exceptions)
        for i in expected_distance:
            assert nearest.nearest_distance[i] == \
                    pytest.approx(expected_distance[i])
        assert nearest.nearest_distance == pytest.approx(expected_distance)

    def test_sorted_distances(self, idx):
        nearest = self.nearest_atoms[idx]
        # TODO: this set should be made global, and everything else is based
        # on it
        expected_items = {
            # these come from test_initialization
            0: [(1, 4, 0.070710678118654752440),
                (4, 6, 0.063639610306789277196),
                (5, 6, 0.045276925690687083132),
                (6, 5, 0.045276925690687083132)],
            4: [(0, 9, 0.045),
                (1, 8, 0.04501110973970759588),
                (4, 6, 0.051),
                (5, 7, 0.051),
                (6, 4, 0.051),
                (7, 5, 0.051),
                (8, 1, 0.04501110973970759588),
                (9, 0, 0.045)]
        }[idx]

        sorted_distances = nearest.sorted_distances
        # test that ordering is correct, then that the items are correct
        approx_expected = [(i[0], i[1], pytest.approx(i[2]))
                           for i in expected_items]

        for expected in approx_expected:
            assert expected in sorted_distances


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
