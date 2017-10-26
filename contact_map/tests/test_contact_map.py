import os
import collections
import numpy as np
import mdtraj as md

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

# includes pytest
from .utils import *

# stuff to be testing in this file
from contact_map.contact_map import *

traj = md.load(find_testfile("trajectory.pdb"))

traj_atom_contact_count = {
    frozenset([0, 8]): 1,
    frozenset([0, 9]): 1,
    frozenset([1, 4]): 4,
    frozenset([1, 5]): 1,
    frozenset([1, 8]): 1,
    frozenset([1, 9]): 1,
    frozenset([4, 6]): 5,
    frozenset([4, 7]): 2,
    frozenset([4, 8]): 1,
    frozenset([5, 6]): 5,
    frozenset([5, 7]): 2,
    frozenset([5, 8]): 1
}

traj_residue_contact_count = {
    frozenset([0, 2]): 5,
    frozenset([0, 4]): 1,
    frozenset([2, 3]): 5,
    frozenset([2, 4]): 1
}

test_file = "test_file.p"

def counter_of_inner_list(ll):
    return collections.Counter([frozenset(i) for i in ll])

def check_most_common_order(most_common):
    for i in range(len(most_common) - 1):
        assert most_common[i][1] >= most_common[i+1][1]

def test_residue_neighborhood():
    top = traj.topology
    residues = list(top.residues)
    for res in residues:
        assert residue_neighborhood(res, n=0) == [res.index]
        for n in range(5):
            from_bottom = -min(0, res.index - n)
            from_top = max(0, res.index + n - (len(residues) - 1))
            len_n = 2*n + 1 - from_top - from_bottom
            assert len(residue_neighborhood(res, n=n)) == len_n


@pytest.mark.parametrize("idx", [0, 4])
class TestContactMap(object):
    def setup(self):
        self.topology = traj.topology
        self.map0 = ContactMap(traj[0], cutoff=0.075, n_neighbors_ignored=0)
        self.map4 = ContactMap(traj[4], cutoff=0.075, n_neighbors_ignored=0)
        self.maps = {0: self.map0, 4: self.map4}

        self.expected_atom_contacts = {
            self.map0: [[1, 4], [4, 6], [5, 6]],
            self.map4: [[0, 9], [0, 8], [1, 8], [1, 9], [1, 4], [8, 4],
                        [8, 5], [4, 6], [4, 7], [5, 6], [5, 7]]
        }

        self.expected_residue_contacts = {
            self.map0: [[0, 2], [2, 3]],
            self.map4: [[0, 2], [2, 3], [0, 4], [2, 4]]
        }

    def test_initialization(self, idx):
        m = self.maps[idx]
        assert set(m.query) == set(range(10))
        assert set(m.haystack) == set(range(10))
        assert m.n_neighbors_ignored == 0
        assert m.topology == self.topology
        for res in m.topology.residues:
            ignored_atoms = m.residue_ignore_atom_idxs[res.index]
            assert ignored_atoms == set([a.index for a in res.atoms])

    def test_counters(self, idx):
        # tests the counters generated in the contact_map method
        m = self.maps[idx]
        expected = counter_of_inner_list(self.expected_atom_contacts[m])
        assert m._atom_contacts == expected
        assert m.atom_contacts.counter == expected
        expected = counter_of_inner_list(self.expected_residue_contacts[m])
        assert m._residue_contacts == expected
        assert m.residue_contacts.counter == expected

    def test_with_ignores(self, idx):
        m = ContactMap(traj[idx], cutoff=0.075, n_neighbors_ignored=1)
        expected_atom_contacts = {
            0: [[1, 4]],
            4: [[0, 9], [0, 8], [1, 8], [1, 9], [1, 4], [8, 4], [8, 5]]
        }
        expected_residue_contacts = {
            0: [[0, 2]],
            4: [[0, 2], [0, 4], [2, 4]]
        }

        expected = counter_of_inner_list(expected_atom_contacts[idx])
        assert m._atom_contacts == expected
        assert m.atom_contacts.counter == expected

        expected = counter_of_inner_list(expected_residue_contacts[idx])
        assert m._residue_contacts == expected
        assert m.residue_contacts.counter == expected

    def test_most_common_atoms_for_residue(self, idx):
        m = self.maps[idx]
        top = self.topology
        expected_atom_indices_for_res_2 = {  # atoms 4, 5
            self.map0: {4: [1, 6], 5: [6]},
            self.map4: {4: [1, 8, 6, 7], 5: [6, 7, 8]}
        }

        most_common_atoms = m.most_common_atoms_for_residue(top.residue(2))
        most_common_idx = m.most_common_atoms_for_residue(2)

        beauty = {frozenset(ll[0]): ll[1] for ll in most_common_atoms}
        beauty_idx = {frozenset(ll[0]): ll[1] for ll in most_common_idx}
        truth = {frozenset([top.atom(k), top.atom(a)]): 1
                 for (k, v) in expected_atom_indices_for_res_2[m].items()
                 for a in v}

        assert beauty == truth
        assert beauty_idx == truth

    def test_most_common_atoms_for_contact(self, idx):
        m = self.maps[idx]
        top = self.topology
        expected_atom_indices_contact_0_2 = {
            self.map0: [[1, 4]],
            self.map4: [[1, 4]]
        }

        contact_pair = [top.residue(0), top.residue(2)]
        most_common_atoms = m.most_common_atoms_for_contact(contact_pair)
        most_common_idx = m.most_common_atoms_for_contact([0, 2])

        beauty = {frozenset(ll[0]): ll[1] for ll in most_common_atoms}
        beauty_idx = {frozenset(ll[0]): ll[1] for ll in most_common_idx}
        truth = {frozenset([top.atom(a) for a in ll]): 1
                 for ll in expected_atom_indices_contact_0_2[m]}

        assert beauty == truth
        assert beauty_idx == truth

    def test_saving(self, idx):
        m = self.maps[idx]
        m.save_to_file(test_file)
        m2 = ContactMap.from_file(test_file)
        assert m.atom_contacts.counter == m2.atom_contacts.counter
        os.remove(test_file)


class TestContactFrequency(object):
    def setup(self):
        self.map = ContactFrequency(trajectory=traj,
                                    cutoff=0.075,
                                    n_neighbors_ignored=0)
        self.expected_atom_contact_count = traj_atom_contact_count
        self.expected_residue_contact_count = traj_residue_contact_count
        self.expected_n_frames = 5

    def test_initialization(self):
        assert self.map.n_frames == len(traj)
        assert self.map.topology == traj.topology
        assert set(self.map.query) == set(range(10))
        assert set(self.map.haystack) == set(range(10))
        assert self.map.n_neighbors_ignored == 0
        for res in self.map.topology.residues:
            ignored_atoms = self.map.residue_ignore_atom_idxs[res.index]
            assert ignored_atoms == set([a.index for a in res.atoms])

    def test_counters(self):
        assert self.map.n_frames == self.expected_n_frames

        atom_contacts = self.map.atom_contacts
        expected_atom_contacts = {
            k: float(v) / self.expected_n_frames
            for (k, v) in self.expected_atom_contact_count.items()
        }
        assert atom_contacts.counter == expected_atom_contacts

        residue_contacts = self.map.residue_contacts
        expected_residue_contacts = {
            k: float(v) / self.expected_n_frames
            for (k, v) in self.expected_residue_contact_count.items()
        }
        assert residue_contacts.counter == expected_residue_contacts

    def test_frames_parameter(self):
        # test that the frames parameter in initialization works
        frames = [1, 3, 4]
        contacts = ContactFrequency(trajectory=traj,
                                    cutoff=0.075,
                                    n_neighbors_ignored=0,
                                    frames=frames)
        expected_atom_raw_count = {
            frozenset([0, 8]): 1,
            frozenset([0, 9]): 1,
            frozenset([1, 4]): 2,
            frozenset([1, 5]): 1,
            frozenset([1, 8]): 1,
            frozenset([1, 9]): 1,
            frozenset([4, 6]): 3,
            frozenset([4, 7]): 2,
            frozenset([4, 8]): 1,
            frozenset([5, 6]): 3,
            frozenset([5, 7]): 2,
            frozenset([5, 8]): 1
        }
        expected_residue_raw_count = {
            frozenset([0, 2]): 3,
            frozenset([0, 4]): 1,
            frozenset([2, 3]): 3,
            frozenset([2, 4]): 1
        }

        expected_atom_count = {
            k: v/3.0 for (k, v) in expected_atom_raw_count.items()
        }
        assert contacts.atom_contacts.counter == expected_atom_count

        expected_residue_count = {
            k: v/3.0 for (k, v) in expected_residue_raw_count.items()
        }
        assert contacts.residue_contacts.counter == expected_residue_count

    def test_saving(self):
        m = self.map
        m.save_to_file(test_file)
        m2 = ContactMap.from_file(test_file)
        assert m.atom_contacts.counter == m2.atom_contacts.counter
        os.remove(test_file)

    @pytest.mark.parametrize('select_by', ['res', 'idx'])
    def test_most_common_atoms_for_residue(self, select_by):
        if select_by == 'res':
            res_2 = self.map.topology.residue(2)
        elif select_by == 'idx':
            res_2 = 2
        else:
            raise RuntimeError("This should not happen")
        # call both by residue and residue number
        most_common_2 = self.map.most_common_atoms_for_residue(res_2)
        check_most_common_order(most_common_2)

        most_common_numbers_2 = {frozenset([k[0].index, k[1].index]): v
                                 for (k, v) in most_common_2}
        # check contents are correct; residue 2 is atoms [4, 5]
        expected_numbers_2 = {
            frozenset([1, 4]): 0.8,
            frozenset([1, 5]): 0.2,
            frozenset([4, 6]): 1.0,
            frozenset([4, 7]): 0.4,
            frozenset([4, 8]): 0.2,
            frozenset([5, 6]): 1.0,
            frozenset([5, 7]): 0.4,
            frozenset([5, 8]): 0.2
        }
        assert most_common_numbers_2 == expected_numbers_2

    @pytest.mark.parametrize('select_by', ['res', 'idx'])
    def test_most_common_atoms_for_contact(self, select_by):
        top = self.map.topology
        if select_by == 'res':
            pair = [top.residue(2), top.residue(3)]
        elif select_by == 'idx':
            pair = [2, 3]
        else:
            raise RuntimeError("This should not happen")

        most_common_2_3 = self.map.most_common_atoms_for_contact(pair)
        check_most_common_order(most_common_2_3)
        most_common_2_3_frozenset = [(frozenset(ll[0]), ll[1])
                                     for ll in most_common_2_3]

        # residue 2: atoms 4, 5; residue 3: atoms 6, 7
        expected_2_3 = [
            (frozenset([top.atom(4), top.atom(6)]), 1.0),
            (frozenset([top.atom(5), top.atom(6)]), 1.0),
            (frozenset([top.atom(4), top.atom(7)]), 0.4),
            (frozenset([top.atom(5), top.atom(7)]), 0.4)
        ]
        assert set(most_common_2_3_frozenset) == set(expected_2_3)


class TestContactCount(object):
    def setup(self):
        self.map = ContactFrequency(traj, cutoff=0.075,
                                    n_neighbors_ignored=0)
        self.topology = self.map.topology
        self.atom_contacts = self.map.atom_contacts
        self.residue_contacts = self.map.residue_contacts

        self.atom_matrix = np.array([
            #  0    1    2    3    4    5    6    7    8    9
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2], # 0
            [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.2, 0.2], # 1
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 2
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # 3
            [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.0], # 4
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.4, 0.2, 0.0], # 5
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], # 6
            [0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0], # 7
            [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0], # 8
            [0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 9
        ])
        self.residue_matrix = np.array([
            #  0    1    2    3    4
            [0.0, 0.0, 1.0, 0.0, 0.2], # 0
            [0.0, 0.0, 0.0, 0.0, 0.0], # 1
            [1.0, 0.0, 0.0, 1.0, 0.2], # 2
            [0.0, 0.0, 1.0, 0.0, 0.0], # 3
            [0.2, 0.0, 0.2, 0.0, 0.0]  # 4
        ])

    # HAS_MATPLOTLIB imported by contact_map wildcard
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Missing matplotlib")
    def test_plot(self):
        # purely smoke test
        self.residue_contacts.plot()
        self.atom_contacts.plot()

    def test_initialization(self):
        assert self.atom_contacts._object_f == self.topology.atom
        assert self.atom_contacts.n_x == self.topology.n_atoms
        assert self.atom_contacts.n_y == self.topology.n_atoms
        assert self.residue_contacts._object_f == self.topology.residue
        assert self.residue_contacts.n_x == self.topology.n_residues
        assert self.residue_contacts.n_y == self.topology.n_residues

    def test_sparse_matrix(self):
        assert_array_equal(self.map.atom_contacts.sparse_matrix.todense(),
                           self.atom_matrix)
        assert_array_equal(self.map.residue_contacts.sparse_matrix.todense(),
                           self.residue_matrix)

    def test_df(self):
        atom_df = self.map.atom_contacts.df
        residue_df = self.map.residue_contacts.df
        assert isinstance(atom_df, pd.SparseDataFrame)
        assert isinstance(residue_df, pd.SparseDataFrame)

        assert_array_equal(atom_df.to_dense().as_matrix(),
                           zero_to_nan(self.atom_matrix))
        assert_array_equal(residue_df.to_dense().as_matrix(),
                           zero_to_nan(self.residue_matrix))

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_most_common(self, obj_type):
        if obj_type == 'atom':
            source_expected = traj_atom_contact_count
            contacts = self.map.atom_contacts
            obj_func = self.topology.atom
        elif obj_type == 'res':
            source_expected = traj_residue_contact_count
            contacts = self.map.residue_contacts
            obj_func = self.topology.residue
        else:
            raise RuntimeError("This shouldn't happen")

        expected = [
            (frozenset([obj_func(idx) for idx in ll[0]]), float(ll[1]) / 5.0)
            for ll in source_expected.items()
        ]

        most_common = contacts.most_common()
        cleaned = [(frozenset(ll[0]), ll[1]) for ll in most_common]

        check_most_common_order(most_common)
        assert set(cleaned) == set(expected)

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_most_common_with_object(self, obj_type):
        top = self.topology
        if obj_type == 'atom':
            contacts = self.map.atom_contacts
            obj = top.atom(4)
            expected = [(frozenset([obj, top.atom(6)]), 1.0),
                        (frozenset([obj, top.atom(1)]), 0.8),
                        (frozenset([obj, top.atom(7)]), 0.4),
                        (frozenset([obj, top.atom(8)]), 0.2)]
        elif obj_type == 'res':
            contacts = self.map.residue_contacts
            obj = self.topology.residue(2)
            expected = [(frozenset([obj, top.residue(0)]), 1.0),
                        (frozenset([obj, top.residue(3)]), 1.0),
                        (frozenset([obj, top.residue(4)]), 0.2)]
        else:
            raise RuntimeError("This shouldn't happen")

        most_common = contacts.most_common(obj)
        cleaned = [(frozenset(ll[0]), ll[1]) for ll in most_common]

        check_most_common_order(most_common)
        assert set(cleaned) == set(expected)

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_most_common_idx(self, obj_type):
        if obj_type == 'atom':
            source_expected = traj_atom_contact_count
            contacts = self.map.atom_contacts
        elif obj_type == 'res':
            source_expected = traj_residue_contact_count
            contacts = self.map.residue_contacts
        else:
            raise RuntimeError("This shouldn't happen")

        expected_count = [(ll[0], float(ll[1]) / 5.0)
                          for ll in source_expected.items()]
        assert set(contacts.most_common_idx()) == set(expected_count)


class TestContactDifference(object):
    def test_diff_traj_frame(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactMap(traj[4], cutoff=0.075, n_neighbors_ignored=0)
        expected_atom_count = {
            frozenset([0, 8]): -1.0,
            frozenset([0, 9]): -1.0,
            frozenset([1, 4]): 0.75 - 1.0,
            frozenset([1, 5]): 0.25,
            frozenset([1, 8]): -1.0,
            frozenset([1, 9]): -1.0,
            frozenset([4, 6]): 0.0,
            frozenset([4, 7]): 0.25 - 1.0,
            frozenset([4, 8]): -1.0,
            frozenset([5, 6]): 0.0,
            frozenset([5, 7]): 0.25 - 1.0,
            frozenset([5, 8]): -1.0
        }
        expected_residue_count = {
            frozenset([0, 2]): 0.0,
            frozenset([0, 4]): -1.0,
            frozenset([2, 3]): 0.0,
            frozenset([2, 4]): -1.0
        }
        diff_1 = ttraj - frame
        diff_2 = frame - ttraj

        assert diff_1.atom_contacts.counter == expected_atom_count
        assert diff_2.atom_contacts.counter == \
                {k: -v for (k, v) in expected_atom_count.items()}

        assert diff_1.residue_contacts.counter == expected_residue_count
        assert diff_2.residue_contacts.counter == \
                {k: -v for (k, v) in expected_residue_count.items()}

    def test_diff_frame_frame(self):
        m0 = ContactMap(traj[0], cutoff=0.075, n_neighbors_ignored=0)
        m4 = ContactMap(traj[4], cutoff=0.075, n_neighbors_ignored=0)
        # one of these simply has more contacts than the other, so to test
        # both positive diff and negative diff we flip the sign
        diff_1 = m4 - m0
        diff_2 = m0 - m4
        # expected diffs are present in m4, not in m0
        expected_atom_diff = [[0, 9], [0, 8], [1, 8], [1, 9], [4, 8],
                              [5, 8], [4, 7], [5, 7]]
        expected_atom_common = [[1, 4], [4, 6], [5, 6]]
        expected_residue_diff = [[0, 4], [2, 4]]
        expected_residue_common = [[0, 2], [2, 3]]

        expected_atoms_1 = counter_of_inner_list(expected_atom_diff)
        # add the zeros
        expected_atoms_1.update({frozenset(pair): 0
                                 for pair in expected_atom_common})
        assert diff_1.atom_contacts.counter == expected_atoms_1
        expected_atoms_2 = {k: -v for (k, v) in expected_atoms_1.items()}
        assert diff_2.atom_contacts.counter == expected_atoms_2

        expected_residues_1 = counter_of_inner_list(expected_residue_diff)
        expected_residues_1.update({frozenset(pair): 0
                                    for pair in expected_residue_common})
        assert diff_1.residue_contacts.counter == expected_residues_1
        expected_residues_2 = {k: -v
                               for (k, v) in expected_residues_1.items()}
        assert diff_2.residue_contacts.counter == expected_residues_2

    def test_diff_traj_traj(self):
        traj_1 = ContactFrequency(trajectory=traj[0:2],
                                  cutoff=0.075,
                                  n_neighbors_ignored=0)
        traj_2 = ContactFrequency(trajectory=traj[3:5],
                                  cutoff=0.075,
                                  n_neighbors_ignored=0)

        expected_atom_count = {
            frozenset([0, 8]): -0.5,
            frozenset([0, 9]): -0.5,
            frozenset([1, 4]): 0.5 - 1.0,
            frozenset([1, 5]): 0.5,
            frozenset([1, 8]): -0.5,
            frozenset([1, 9]): -0.5,
            frozenset([4, 6]): 0.0,
            frozenset([4, 7]): -1.0,
            frozenset([4, 8]): -0.5,
            frozenset([5, 6]): 0.0,
            frozenset([5, 7]): -1.0,
            frozenset([5, 8]): -0.5
        }
        expected_residue_count = {
            frozenset([0, 2]): 0,
            frozenset([0, 4]): -0.5,
            frozenset([2, 3]): 0,
            frozenset([2, 4]): -0.5
        }

        diff_1 = traj_1 - traj_2
        diff_2 = traj_2 - traj_1

        assert diff_1.atom_contacts.counter == expected_atom_count
        assert diff_2.atom_contacts.counter == \
                {k: -v for (k, v) in expected_atom_count.items()}

        assert diff_1.residue_contacts.counter == expected_residue_count
        assert diff_2.residue_contacts.counter == \
                {k: -v for (k, v) in expected_residue_count.items()}

    def test_saving(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactMap(traj[4], cutoff=0.075, n_neighbors_ignored=0)
        diff = ttraj - frame

        diff.save_to_file(test_file)
        reloaded = ContactDifference.from_file(test_file)
        assert diff.atom_contacts.counter == reloaded.atom_contacts.counter
        os.remove(test_file)

