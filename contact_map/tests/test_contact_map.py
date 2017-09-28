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


class TestContactMap(object):
    def setup(self):
        self.map0 = ContactMap(traj[0], cutoff=0.075, n_neighbors_ignored=0)
        self.map4 = ContactMap(traj[4], cutoff=0.075, n_neighbors_ignored=0)

    def test_setup(self):
        for m in [self.map0, self.map4]:
            assert set(m.query) == set(range(10))
            assert set(m.haystack) == set(range(10))
            assert m.n_neighbors_ignored == 0
            for res in m.topology.residues:
                ignored_atoms = m.residue_ignore_atom_idxs[res.index]
                assert ignored_atoms == set([a.index for a in res.atoms])

    def test_contact_map(self):
        # this tests the internal structure contact map object; note that
        # this only underscored attributes, so this test can change
        expected_atom_contacts = {
            self.map0: [[1, 4], [4, 6], [5, 6]],
            self.map4: [[0, 9], [0, 8], [1, 8], [1, 9], [1, 4], [8, 4],
                        [8, 5], [4, 6], [4, 7], [5, 6], [5, 7]]
        }

        expected_residue_contacts = {
            self.map0: [[0, 2], [2, 3]],
            self.map4: [[0, 2], [2, 3], [0, 4], [2, 4]]
        }
        for m, contacts in expected_atom_contacts.items():
            expected = collections.Counter([frozenset(c) for c in contacts])
            assert m._atom_contacts == expected
        for m, contacts in expected_residue_contacts.items():
            expected = collections.Counter([frozenset(c) for c in contacts])
            assert m._residue_contacts == expected

    def test_atom_contacts(self):
        pytest.skip()

    def test_residue_contacts(self):
        pytest.skip()



class TestContactFrequency(object):
    pass
