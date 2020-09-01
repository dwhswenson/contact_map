# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *
from .test_contact_map import (
    counter_of_inner_list, _contact_object_compare, traj_atom_contact_count,
    traj_residue_contact_count
)

import mdtraj as md

from contact_map.contact_trajectory import *
from contact_map.contact_count import ContactCount

TRAJ_ATOM_CONTACTS = [
    [[1, 4], [4, 6], [5, 6]],
    [[1, 5], [4, 6], [5, 6]],
    [[1, 4], [4, 6], [5, 6]],
    [[1, 4], [4, 6], [4, 7], [5, 6], [5, 7]],
    [[0, 9], [0, 8], [1, 8], [1, 9], [1, 4], [8, 4], [8, 5], [4, 6], [4, 7],
     [5, 6], [5, 7]]
]

TRAJ_RES_CONTACTS = [
    [[0, 2], [2, 3]],
    [[0, 2], [2, 3]],
    [[0, 2], [2, 3]],
    [[0, 2], [2, 3]],
    [[0, 2], [2, 3], [0, 4], [2, 4]]
]

class TestContactTrajectory(object):
    def setup(self):
        self.traj = md.load(find_testfile("trajectory.pdb"))
        self.map = ContactTrajectory(self.traj, cutoff=0.075,
                                     n_neighbors_ignored=0)
        self.expected_atom_contacts = TRAJ_ATOM_CONTACTS
        self.expected_residue_contacts = TRAJ_RES_CONTACTS

    @pytest.mark.parametrize('contact_type', ['atom', 'residue'])
    def test_contacts(self, contact_type):
        assert len(self.map) == 5
        contacts = {'atom': self.map.atom_contacts,
                    'residue': self.map.residue_contacts}[contact_type]
        expected = {'atom': self.expected_atom_contacts,
                    'residue': self.expected_residue_contacts}[contact_type]

        for contact, expect in zip(contacts, expected):
            expected_counter = counter_of_inner_list(expect)
            assert contact.counter == expected_counter

    @pytest.mark.parametrize('contact_type', ['atom', 'residue'])
    def test_contacts_sliced(self, contact_type):
        selected_atoms = [2, 3, 4, 5, 6, 7, 8, 9]
        cmap = ContactTrajectory(self.traj, query=selected_atoms,
                                 haystack=selected_atoms, cutoff=0.075,
                                 n_neighbors_ignored=0)
        contacts = {'atom': cmap.atom_contacts,
                    'residue': cmap.residue_contacts}[contact_type]
        expected = {
            'atom': [
                [[4, 6], [5, 6]],
                [[4, 6], [5, 6]],
                [[4, 6], [5, 6]],
                [[4, 6], [4, 7], [5, 6], [5, 7]],
                [[8, 4], [8, 5], [4, 6], [4, 7], [5, 6], [5, 7]]
            ],
            'residue': [
                [[2, 3]],
                [[2, 3]],
                [[2, 3]],
                [[2, 3]],
                [[2, 3], [2, 4]]
            ]
        }[contact_type]

        for contact, expect in zip(contacts, expected):
            expected_counter = counter_of_inner_list(expect)
            assert contact.counter == expected_counter


    @pytest.mark.parametrize('contactcount', [True, False])
    def test_from_contacts(self, contactcount):
        atom_contacts = [
            counter_of_inner_list(frame_contacts)
            for frame_contacts in self.expected_atom_contacts
        ]
        residue_contacts = [
            counter_of_inner_list(frame_contacts)
            for frame_contacts in self.expected_residue_contacts
        ]
        top = self.traj.topology
        if contactcount:
            atom_contacts = [ContactCount(contact, top.atom, 10, 10)
                             for contact in atom_contacts]
            residue_contacts = [ContactCount(contact, top.residue, 5, 5)
                                for contact in residue_contacts]

        cmap = ContactTrajectory.from_contacts(atom_contacts,
                                               residue_contacts,
                                               topology=top,
                                               cutoff=0.075,
                                               n_neighbors_ignored=0)
        for truth, beauty in zip(self.map, cmap):
            _contact_object_compare(truth, beauty)
            assert truth == beauty
        _contact_object_compare(cmap, self.map)
        assert cmap == self.map


    def test_contact_frequency(self):
        freq = self.map.contact_frequency()
        expected_atom_count = {
            key: val / 5.0 for key, val in traj_atom_contact_count.items()
        }
        expected_res_count = {
            key: val / 5.0
            for key, val in traj_residue_contact_count.items()
        }
        assert freq.atom_contacts.counter == expected_atom_count
        assert freq.residue_contacts.counter == expected_res_count

    @pytest.mark.parametrize("intermediate", ["dict", "json"])
    def test_serialization_cycle(self, intermediate):
        # NOTE: this is identical to TestContactFrequency; can probably
        # abstract it out
        serializer, deserializer = {
            'json': (self.map.to_json, ContactTrajectory.from_json),
            'dict': (self.map.to_dict, ContactTrajectory.from_dict)
        }[intermediate]

        serialized = serializer()
        reloaded = deserializer(serialized)
        _contact_object_compare(self.map, reloaded)
        assert self.map == reloaded

    def test_from_contact_maps(self):
        maps = [ContactFrequency(frame, cutoff=0.075, n_neighbors_ignored=0)
                for frame in self.traj]
        cmap = ContactTrajectory.from_contact_maps(maps)
        _contact_object_compare(self.map, cmap)
        assert self.map == cmap

    def test_from_contact_maps_incompatible(self):
        map0 = ContactFrequency(self.traj[0], cutoff=0.075,
                                n_neighbors_ignored=0)
        maps = [map0] + [ContactFrequency(frame) for frame in self.traj[1:]]
        with pytest.raises(AssertionError):
            _ = ContactTrajectory.from_contact_maps(maps)

    def test_join(self):
        segments = self.traj[0], self.traj[1:3], self.traj[3:]
        assert [len(s) for s in segments] == [1, 2, 2]
        assert md.join(segments) == self.traj

        cmaps = [ContactTrajectory(segment, cutoff=0.075,
                                   n_neighbors_ignored=0)
                 for segment in segments]

        cmap = ContactTrajectory.join(cmaps)

        assert len(cmap) == len(self.map)
        for i, (truth, beauty) in enumerate(zip(self.map, cmap)):
            _contact_object_compare(truth, beauty)
            assert truth == beauty

        _contact_object_compare(self.map, cmap)
        assert self.map == cmap

    def test_rolling_frequency(self):
        # smoke test; correctness is tested in tests for
        # RollingContactFrequency
        assert len(list(self.map.rolling_frequency(window_size=2))) == 4


class TestMutableContactTrajectory(object):
    def setup(self):
        self.traj = md.load(find_testfile("trajectory.pdb"))
        self.map = MutableContactTrajectory(self.traj, cutoff=0.075,
                                            n_neighbors_ignored=0)
        self.expected_atom_contacts = TRAJ_ATOM_CONTACTS.copy()
        self.expected_residue_contacts = TRAJ_RES_CONTACTS.copy()

    def _test_expected_contacts(self, traj_map, exp_atoms, exp_res):
        for cmap, exp_a, exp_r in zip(traj_map, exp_atoms, exp_res):
            atom_counter = cmap.atom_contacts.counter
            res_counter = cmap.residue_contacts.counter
            assert atom_counter == counter_of_inner_list(exp_a)
            assert res_counter == counter_of_inner_list(exp_r)

    def test_setitem(self):
        cmap4 = ContactFrequency(self.traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        self.map[1] = cmap4
        expected_atoms = self.expected_atom_contacts
        expected_atoms[1] = expected_atoms[4]
        expected_res = self.expected_residue_contacts
        expected_res[1] = expected_res[4]
        self._test_expected_contacts(self.map, expected_atoms, expected_res)

    def test_delitem(self):
        del self.map[1]
        assert len(self.map) == 4
        expected_atoms = (self.expected_atom_contacts[:1]
                          + self.expected_atom_contacts[2:])
        expected_res = (self.expected_residue_contacts[:1]
                        + self.expected_residue_contacts[2:])
        self._test_expected_contacts(self.map, expected_atoms, expected_res)

    def test_insert(self):
        cmap4 = self.map[4]
        self.map.insert(0, cmap4)
        expected_atoms = [TRAJ_ATOM_CONTACTS[4]] + TRAJ_ATOM_CONTACTS
        expected_res = [TRAJ_RES_CONTACTS[4]] + TRAJ_RES_CONTACTS
        self._test_expected_contacts(self.map, expected_atoms, expected_res)

    def test_hash_eq(self):
        cmap = MutableContactTrajectory(self.traj, cutoff=0.075,
                                        n_neighbors_ignored=0)
        assert hash(cmap) != hash(self.map)
        assert cmap != self.map


class TestWindowedIterator(object):
    def setup(self):
        self.iter = WindowedIterator(length=10, width=3, step=2,
                                     slow_build=False)

    def test_startup_normal(self):
        to_add, to_sub = self.iter._startup()
        assert to_sub == slice(0, 0)
        assert to_add == slice(0, 3)
        assert self.iter.min == 0
        assert self.iter.max == 2

    def test_startup_slow_build_step1(self):
        itr = WindowedIterator(length=10, width=3, step=1, slow_build=True)
        to_add, to_sub = itr._startup()
        assert to_sub == slice(0, 0)
        assert to_add == slice(0, 1)
        assert itr.min == 0
        assert itr.max == 0

        to_add, to_sub = itr._startup()
        assert to_sub == slice(0, 0)
        assert to_add == slice(1, 2)
        assert itr.min == 0
        assert itr.max == 1

    def test_normal(self):
        self.iter.min = 0
        self.iter.max = 2
        to_add, to_sub = self.iter._normal()
        assert to_sub == slice(0, 2)
        assert to_add == slice(3, 5)
        assert self.iter.min == 2
        assert self.iter.max == 4

    @pytest.mark.parametrize('length,width,step,slow_build,expected', [
        (5, 3, 2, False, [(slice(0, 0), slice(0, 3), 0, 2),
                           (slice(0, 2), slice(3, 5), 2, 4)]),
        (5, 3, 1, True, [(slice(0, 0), slice(0, 1), 0, 0),
                         (slice(0, 0), slice(1, 2), 0, 1),
                         (slice(0, 0), slice(2, 3), 0, 2),
                         (slice(0, 1), slice(3, 4), 1, 3),
                         (slice(1, 2), slice(4, 5), 2, 4)]),
        (5, 3, 2, True, [(slice(0, 0), slice(0, 2), 0, 1),
                         (slice(0, 1), slice(2, 4), 1, 3)]),
        (6, 3, 3, False, [(slice(0, 0), slice(0, 3), 0, 2),
                          (slice(0, 3), slice(3, 6), 3, 5)]),
        (6, 3, 3, True, [(slice(0, 0), slice(0, 3), 0, 2),
                         (slice(0, 3), slice(3, 6), 3, 5)]),
    ])
    def test_next(self, length, width, step, slow_build, expected):
        itr = WindowedIterator(length, width, step, slow_build)
        for expect in expected:
            exp_sub, exp_add, exp_min, exp_max = expect
            to_add, to_sub = next(itr)
            assert to_add == exp_add
            assert to_sub == exp_sub
            assert itr.min == exp_min
            assert itr.max == exp_max
        with pytest.raises(StopIteration):
            next(itr)


class TestRollingContactFrequency(object):
    def setup(self):
        self.traj = md.load(find_testfile("trajectory.pdb"))
        self.map = ContactTrajectory(self.traj, cutoff=0.075,
                                     n_neighbors_ignored=0)
        self.rolling_freq = RollingContactFrequency(self.map, width=2,
                                                    step=1)
        self.expected_atoms = [
            {frozenset([1, 4]): 0.5, frozenset([4, 6]): 1.0,
             frozenset([5, 6]): 1.0, frozenset([1, 5]): 0.5},
            {frozenset([1, 5]): 0.5, frozenset([4, 6]): 1.0,
             frozenset([5, 6]): 1.0, frozenset([1, 4]): 0.5},
            {frozenset([1, 4]): 1.0, frozenset([4, 6]): 1.0,
             frozenset([5, 6]): 1.0, frozenset([4, 7]): 0.5,
             frozenset([5, 7]): 0.5},
            {frozenset([0, 9]): 0.5, frozenset([0, 8]): 0.5,
             frozenset([1, 8]): 0.5, frozenset([1, 9]): 0.5,
             frozenset([1, 4]): 1.0, frozenset([8, 4]): 0.5,
             frozenset([8, 5]): 0.5, frozenset([4, 6]): 1.0,
             frozenset([4, 7]): 1.0, frozenset([5, 6]): 1.0,
             frozenset([5, 7]): 1.0}
        ]
        self.expected_residues = [
            {frozenset([0, 2]): 1.0, frozenset([2, 3]): 1.0},
            {frozenset([0, 2]): 1.0, frozenset([2, 3]): 1.0},
            {frozenset([0, 2]): 1.0, frozenset([2, 3]): 1.0},
            {frozenset([0, 2]): 1.0, frozenset([2, 3]): 1.0,
             frozenset([0, 4]): 0.5, frozenset([2, 4]): 0.5}
        ]

    def test_normal_iteration(self):
        results = list(freq for freq in self.rolling_freq)
        assert len(results) == 4

        atom_contacts = [r.atom_contacts.counter for r in results]
        for beauty, truth in zip(atom_contacts, self.expected_atoms):
            assert beauty == truth

        residue_contacts = [r.residue_contacts.counter for r in results]
        for beauty, truth in zip(residue_contacts, self.expected_residues):
            assert beauty == truth

    def test_slow_build_iteration(self):
        self.rolling_freq.slow_build_iter = True
        results = list(freq for freq in self.rolling_freq)
        assert len(results) == 5

        expected_atoms = [{frozenset([1, 4]): 1.0, frozenset([4, 6]): 1.0,
                           frozenset([5, 6]): 1.0}] + self.expected_atoms
        expected_residues = (
            [{frozenset([0, 2]): 1.0, frozenset([2, 3]): 1.0}]
            + self.expected_residues
        )

        atom_contacts = [r.atom_contacts.counter for r in results]
        for beauty, truth in zip(atom_contacts, expected_atoms):
            assert beauty == truth

        residue_contacts = [r.residue_contacts.counter for r in results]
        for beauty, truth in zip(residue_contacts, expected_residues):
            assert beauty == truth
