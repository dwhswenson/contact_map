import pytest
import mdtraj as md
import collections
import itertools
from contact_map.contact_map import ContactMap, residue_neighborhood

traj = md.load("./trajectory.pdb")

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

def counter_of_inner_list(ll):
    return collections.Counter([frozenset(i) for i in ll])

def index_pairs_to_atom(ll, topology):
    return [frozenset([topology.atom(j) for j in i]) for i in ll]

def index_pairs_to_residue(ll, topology):
    return [frozenset([topology.residue(j) for j in i]) for i in ll]

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

    def test_setup(self, idx):
        m = self.maps[idx]
        assert set(m.query) == set(range(10))
        assert set(m.haystack) == set(range(10))
        assert m.n_neighbors_ignored == 0
        assert m.topology == self.topology
        for res in m.topology.residues:
            ignored_atoms = m.residue_ignore_atom_idxs[res.index]
            assert ignored_atoms == set([a.index for a in res.atoms])

    def test_contact_map(self, idx):
        # this tests the internal structure contact map object; note that
        # this only underscored attributes, so this test can change
        m = self.maps[idx]
        expected = counter_of_inner_list(self.expected_atom_contacts[m])
        assert m._atom_contacts == expected
        assert m.atom_contacts.counter == expected
        expected = counter_of_inner_list(self.expected_residue_contacts[m])
        assert m._residue_contacts == expected
        assert m.residue_contacts.counter == expected

    def test_with_ignores(self, idx):
        pytest.skip()

    def test_most_common_atoms_for_residue(self, idx):
        pytest.skip()

    def test_most_common_atoms_for_contact(self, idx):
        pytest.skip()

    def test_saving(self, idx):
        pytest.skip()


class TestContactCount(object):
    def setup(self):
        self.map = ContactMap(traj[0], cutoff=0.075, n_neighbors_ignored=0)
        self.topology = self.map.topology
        self.atom_contacts = self.map.atom_contacts
        self.residue_contacts = self.map.residue_contacts

    def test_initialization(self):
        assert self.atom_contacts._object_f == self.topology.atom
        assert self.atom_contacts.n_x == self.topology.n_atoms
        assert self.atom_contacts.n_y == self.topology.n_atoms
        assert self.residue_contacts._object_f == self.topology.residue
        assert self.residue_contacts.n_x == self.topology.n_residues
        assert self.residue_contacts.n_y == self.topology.n_residues

    def test_sparse_matrix(self):
        pytest.skip()

    def test_df(self):
        pytest.skip()

    def test_most_common(self):
        pytest.skip()

    def test_most_common_idx(self):
        pytest.skip()




class TestContactFrequency(object):
    pass
