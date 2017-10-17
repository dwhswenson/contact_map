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
            4: {0: 0.047,
                1: 0.04701063709417263364,  # 0.1*sqrt(0.47^2 + 0.01^2)
                4: 0.051,
                5: 0.052,
                6: 0.051,
                7: 0.052,
                8: 0.04701063709417263364,
                9: 0.047}
        }[idx]
        assert nearest.nearest == expected_nearest_atoms
        assert nearest.nearest_distance == pytest.approx(expected_distance)

    def test_with_all_atoms(self, idx):
        n_atoms = traj.n_atoms
        distance_exceptions = {
            0: {5: 0.045276925690687083132,  # 0.1*sqrt(0.05^2 + 0.45^2)
                6: 0.045276925690687083132,
                7: 0.05522680508593630387},  # 0.1*sqrt(0.05^2 + 0.55^2)
            4: {0: 0.047,
                1: 0.04701063709417263364,  # 0.1*sqrt(0.47^2 + 0.01^2
                6: 0.05000999900019995001,  # 0.1*sqrt(0.50^2 + 0.01^2)
                7: 0.05000999900019995001,  # 0.1*sqrt(0.50^2 + 0.01^2)
                8: 0.04701063709417263364,  # 0.1*sqrt(0.47^2 + 0.01^2
                9: 0.047}
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
            4: [(0, 9, 0.047),
                (1, 8, 0.04701063709417263364),
                (4, 6, 0.051),
                (5, 7, 0.052),
                (6, 4, 0.051),
                (7, 5, 0.052),
                (8, 1, 0.04701063709417263364),
                (9, 0, 0.047)]
        }[idx]

        sorted_distances = nearest.sorted_distances
        # test that ordering is correct, then that the items are correct
        approx_expected = [(i[0], i[1], pytest.approx(i[2]))
                           for i in expected_items]

        for expected in approx_expected:
            assert expected in sorted_distances


class TestMinimumDistanceCounter(object):
    def setup(self):
        self.topology = traj.topology
        query = [4, 5]
        haystack = list(set(range(10)) - set(query))
        self.min_dist = MinimumDistanceCounter(traj, query, haystack)
        self.expected_atom_history_idx = [(5, 6), (4, 6), (5, 6), (5, 7),
                                          (4, 6)]
        self.expected_residue_history_idx = [(2, 3), (2, 3), (2, 3), (2, 3),
                                             (2, 3)]

    def test_initialization(self):
        assert traj.topology == self.min_dist.topology
        expected_atom_pairs = [(4, 0), (4, 1), (4, 2), (4, 3),
                               (4, 6), (4, 7), (4, 8), (4, 9),
                               (5, 0), (5, 1), (5, 2), (5, 3),
                               (5, 6), (5, 7), (5, 8), (5, 9)]
        expected_atom_sets = [frozenset(p) for p in expected_atom_pairs]
        atom_sets = [frozenset(p) for p in self.min_dist.atom_pairs]
        assert set(atom_sets) == set(expected_atom_sets)
        expected_min_dist = pytest.approx([
            0.045276925690687083132,
            0.045276925690687083132,
            0.045276925690687083132,
            0.05,
            0.051
        ])
        # In Py-3.4 (at least on Travis), the equality returned a numpy
        # array, leading to a need for .all(). Elsewhere it returned a bool.
        # Forcing minimum_distances to list should fix.
        minimum_distances = self.min_dist.minimum_distances.tolist()
        assert minimum_distances == expected_min_dist

    @staticmethod
    def _pairs_to_frozensets(pairs, convert=None):
        if convert is None:
            convert = lambda x: x
        return [frozenset([convert(p) for p in pair]) for pair in pairs]

    def test_atom_history(self):
        expected_history = self._pairs_to_frozensets(
            pairs=self.expected_atom_history_idx,
            convert=self.topology.atom
        )
        history = self._pairs_to_frozensets(self.min_dist.atom_history)
        assert history == expected_history

    def test_atom_count(self):
        expected_atom_idx_count = {(5, 6): 2, (5, 7): 1, (4, 6): 2}
        expected_atom_count = {
            self._pairs_to_frozensets(
                pairs=[k],
                convert=self.topology.atom
            )[0]: expected_atom_idx_count[k]
            for k in expected_atom_idx_count
        }
        actual_atom_count = {frozenset(k): v
                             for (k, v) in self.min_dist.atom_count.items()}
        assert actual_atom_count == expected_atom_count

    def test_residue_history(self):
        expected_history = self._pairs_to_frozensets(
            pairs=self.expected_residue_history_idx,
            convert=self.topology.residue
        )
        history = self._pairs_to_frozensets(self.min_dist.residue_history)
        assert history == expected_history

    def test_residue_count(self):
        convert = self.topology.residue
        expected_count = {frozenset([convert(2), convert(3)]): 5}
        actual_count = {frozenset(k): v
                        for (k, v) in self.min_dist.residue_count.items()}
        assert actual_count == expected_count
