# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *
from .test_contact_map import counter_of_inner_list, _contact_object_compare

import mdtraj as md

from contact_map.contact_trajectory import *
from contact_map.contact_count import ContactCount

class TestContactTrajectory(object):
    def setup(self):
        self.traj = md.load(find_testfile("trajectory.pdb"))
        self.map = ContactTrajectory(self.traj, cutoff=0.075,
                                     n_neighbors_ignored=0)
        self.expected_atom_contacts = [
            [[1, 4], [4, 6], [5, 6]],
            [[1, 5], [4, 6], [5, 6]],
            [[1, 4], [4, 6], [5, 6]],
            [[1, 4], [4, 6], [4, 7], [5, 6], [5, 7]],
            [[0, 9], [0, 8], [1, 8], [1, 9], [1, 4], [8, 4], [8, 5], [4, 6],
             [4, 7], [5, 6], [5, 7]]
        ]
        self.expected_residue_contacts = [
            [[0, 2], [2, 3]],
            [[0, 2], [2, 3]],
            [[0, 2], [2, 3]],
            [[0, 2], [2, 3]],
            [[0, 2], [2, 3], [0, 4], [2, 4]]
        ]

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


class TestContactTrajectoryWindow(object):
    def setup(self):
        pass

    def test_normal_iteration(self):
        pytest.skip()
        pass

    def test_slow_build_iteration(self):
        pytest.skip()
        pass
