import numpy as np

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

# includes pytest
from .utils import *
from contact_map.contact_map import ContactFrequency
from .test_contact_map import (traj, traj_atom_contact_count,
                               traj_residue_contact_count,
                               check_most_common_order)

from contact_map.contact_count import *

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
        # purely smoke tests
        self.residue_contacts.plot()
        self.atom_contacts.plot()
        self.residue_contacts.plot(with_colorbar=False)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Missing matplotlib")
    def test_plot_kwargs(self):
        fig, _ = self.residue_contacts.plot(figsize=(12, 13), dpi=192)
        # Assert that the kwargs have been passed through
        assert fig.get_dpi() == 192
        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 13

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Missing matplotlib")
    def test_pixel_warning(self):
        # This should not raise a warning (5*2>=10)
        with pytest.warns(None) as record:
            self.atom_contacts.plot(figsize=(5, 5), dpi=2)
        # See if no warning was raised
        assert len(record) == 0

        # Now raise the warning as 4*2 < 10
        with pytest.warns(RuntimeWarning) as record:
            self.atom_contacts.plot(figsize=(4, 4), dpi=2)
        assert len(record) == 1

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

        assert_array_equal(atom_df.to_dense().values,
                           zero_to_nan(self.atom_matrix))
        assert_array_equal(residue_df.to_dense().values,
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
