import os
import collections
import mdtraj as md
import copy

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

# includes pytest
from .utils import *

# stuff to be testing in this file
from contact_map.contact_map import *
from contact_map.contact_count import HAS_MATPLOTLIB, ContactCount

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


def pdb_topology_dict():
    serial = {str(i): i+1 for i in range(10)}
    name = {str(i): "C" + str(i % 2 + 1) for i in range(10)}
    element = {str(i): "C" for i in range(10)}
    res_seq = {str(i): str(i/2 + 1) for i in range(10)}
    res_name = {str(i): "XXX" for i in range(10)}
    chain_id = {str(i): 0 for i in range(10)}
    seg_id = {str(i): "" for i in range(10)}
    dct = {'serial': serial,
           'name': name,
           'element': element,
           'resSeq': res_seq,
           'resName': res_name,
           'chainID': chain_id,
           'segmentID': seg_id}
    return dct


def counter_of_inner_list(ll):
    return collections.Counter([frozenset(i) for i in ll])


def check_most_common_order(most_common):
    for i in range(len(most_common) - 1):
        assert most_common[i][1] >= most_common[i+1][1]


def check_use_atom_slice(m, use_atom_slice, expected):
    if use_atom_slice is not None:
        assert m._use_atom_slice == use_atom_slice
    else:
        assert m._use_atom_slice == expected[m]


def _contact_object_compare(m, m2):
    """Compare two contact objects (with asserts).

    May later become pytest fanciness (pytest_assertrepr_compare)
    """
    assert m.cutoff == m2.cutoff
    assert m.query == m2.query
    assert m.haystack == m2.haystack
    assert m.n_neighbors_ignored == m2.n_neighbors_ignored
    assert m.topology == m2.topology
    if hasattr(m, '_atom_contacts') or hasattr(m2, '_atom_contacts'):
        assert m._atom_contacts == m2._atom_contacts
    if hasattr(m, '_residue_contacts') or hasattr(m2, '_residue_contacts'):
        assert m._residue_contacts == m2._residue_contacts


def _check_contacts_dict_names(contact_object):
    aliases = {
        contact_object.residue_contacts: ['residue', 'residues', 'res'],
        contact_object.atom_contacts: ['atom', 'atoms']
    }
    for (contacts, names) in aliases.items():
        for name in names:
            assert contacts.counter == contact_object.contacts[name].counter


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
class TestContactObject(object):
    # note: these used to be the tests for the separate single-frame
    # ContactMap class; however, it includes a lot of good unit tests for
    # ContactObject
    def setup_method(self):
        self.topology = traj.topology
        self.map0 = ContactFrequency(traj[0], cutoff=0.075,
                                     n_neighbors_ignored=0)
        self.map4 = ContactFrequency(traj[4], cutoff=0.075,
                                     n_neighbors_ignored=0)
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
        assert set(m.all_atoms) == set(range(10))
        assert set(r.index for r in m.query_residues) == set(range(5))
        assert set(r.index for r in m.haystack_residues) == set(range(5))
        assert m.haystack_residue_range == (0, 5)
        assert m.query_residue_range == (0, 5)
        assert m.n_neighbors_ignored == 0
        assert m.topology == self.topology
        for res in m.topology.residues:
            ignored_atoms = m._residue_ignore_atom_idxs[res.index]
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

    def test_to_dict(self, idx):
        m = self.maps[idx]
        dct = m.to_dict()
        # NOTE: topology only tested in a cycle; JSON order not guaranteed
        assert dct['cutoff'] == 0.075
        assert dct['query'] == list(range(10))
        assert dct['haystack'] == list(range(10))
        assert dct['all_atoms'] == tuple(range(10))
        assert dct['n_neighbors_ignored'] == 0

    def test_topology_serialization_cycle(self, idx):
        top = self.maps[idx].topology
        serialized_topology = ContactFrequency._serialize_topology(top)
        new_top = ContactFrequency._deserialize_topology(serialized_topology)
        assert top == new_top

    def test_counter_serialization_cycle(self, idx):
        m = self.maps[idx]
        serialize = ContactFrequency._serialize_contact_counter
        deserialize = ContactFrequency._deserialize_contact_counter
        serialized_atom_counter = serialize(m._atom_contacts)
        serialized_residue_counter = serialize(m._residue_contacts)
        new_atom_counter = deserialize(serialized_atom_counter)
        new_residue_counter = deserialize(serialized_residue_counter)
        assert new_atom_counter == m._atom_contacts
        assert new_residue_counter == m._residue_contacts

    def test_dict_serialization_cycle(self, idx):
        m = self.maps[idx]
        dct = m.to_dict()
        m2 = ContactFrequency.from_dict(dct)
        _contact_object_compare(m, m2)
        assert m == m2

    def test_json_serialization_cycle(self, idx):
        m = self.maps[idx]
        json_str = m.to_json()
        m2 = ContactFrequency.from_json(json_str)
        _contact_object_compare(m, m2)
        assert m == m2

    def test_with_ignores(self, idx):
        m = ContactFrequency(traj[idx], cutoff=0.075, n_neighbors_ignored=1)
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
        m2 = ContactFrequency.from_file(test_file)
        assert m.atom_contacts.counter == m2.atom_contacts.counter
        os.remove(test_file)

    @pytest.mark.parametrize("use_atom_slice", [True, False, None])
    def test_atom_slice(self, idx, use_atom_slice):
        # Set class variable before init
        class_default = ContactFrequency._class_use_atom_slice
        ContactFrequency._class_use_atom_slice = use_atom_slice
        map0q = ContactFrequency(traj[0], query=[1, 4, 5, 6],  cutoff=0.075,
                                 n_neighbors_ignored=0)
        map0h = ContactFrequency(traj[0], haystack=[1, 4, 5, 6],
                                 cutoff=0.075, n_neighbors_ignored=0)
        map0b = ContactFrequency(traj[0], query=[1, 4, 5, 6],
                                 haystack=[1, 4, 5, 6], cutoff=0.075,
                                 n_neighbors_ignored=0)
        maps = [map0q, map0h, map0b]
        atoms = {map0q: list(range(10)),
                 map0h: list(range(10)),
                 map0b: [1, 4, 5, 6]}
        expected_atom_slice = {map0q: False,
                               map0h: False,
                               map0b: True}
        # Only test for map 0 for now
        m0 = self.maps[0]

        # Test init
        for m in maps:
            assert m.all_atoms == atoms[m]
            check_use_atom_slice(m, use_atom_slice, expected_atom_slice)

            # Test results compared to m0
            expected = counter_of_inner_list(self.expected_atom_contacts[m0])
            assert m._atom_contacts == expected
            assert m.atom_contacts.counter == expected
            expected_residue_contacts = self.expected_residue_contacts[m0]
            expected = counter_of_inner_list(expected_residue_contacts)
            assert m._residue_contacts == expected
            assert m.residue_contacts.counter == expected

        # Test sliced indices
        sliced_idx = [0, 1, 2, 3]
        real_idx = [map0b.indexer.real_idx[i] for i in sliced_idx]
        if map0b._use_atom_slice:
            assert real_idx == [1, 4, 5, 6]
        else:
            assert real_idx == sliced_idx
        # Reset class variable (as imports are not redone between function
        # calls)
        ContactFrequency._class_use_atom_slice = class_default

    def test_contacts_dict(self, idx):
        _check_contacts_dict_names(self.maps[idx])

    def test_no_unitcell(self, idx):
        temptraj = copy.deepcopy(traj)
        # Strip unitcell
        temptraj.unitcell_vectors = None

        # Activate atom_slice
        atoms = [1, 4, 5, 6]
        mapi = ContactFrequency(temptraj[idx], cutoff=0.075,
                                n_neighbors_ignored=0, query=atoms,
                                haystack=atoms)
        expected_atom_contacts = {0: [[1, 4], [4, 6], [5, 6]],
                                  4: [[1, 4], [4, 6], [5, 6]]}
        expected = counter_of_inner_list(expected_atom_contacts[idx])
        assert mapi._atom_contacts == expected
    # TODO: add tests for ContactObject._check_consistency


class TestContactFrequency(object):
    def setup_method(self):
        self.atoms = [0, 1, 4, 5, 6, 7, 8, 9]
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
        assert set(self.map.all_atoms) == set(range(10))
        assert self.map.n_neighbors_ignored == 0
        for res in self.map.topology.residues:
            ignored_atoms = self.map._residue_ignore_atom_idxs[res.index]
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

    def test_contacts_dict(self):
        _check_contacts_dict_names(self.map)

    @pytest.mark.parametrize('contactcount', [True, False])
    def test_from_contacts(self, contactcount):
        atom_contacts = self.expected_atom_contact_count
        residue_contacts = self.expected_residue_contact_count
        top = traj.topology
        if contactcount:
            atom_contacts = ContactCount(atom_contacts, top.atom,
                                         10, 10)
            residue_contacts = ContactCount(residue_contacts, top.residue,
                                            5, 5)

        cmap = ContactFrequency.from_contacts(atom_contacts,
                                              residue_contacts,
                                              n_frames=5,
                                              topology=top,
                                              cutoff=0.075,
                                              n_neighbors_ignored=0)
        _contact_object_compare(cmap, self.map)

    def test_check_compatibility_true(self):
        map2 = ContactFrequency(trajectory=traj[0:2],
                                cutoff=0.075,
                                n_neighbors_ignored=0)
        assert self.map._check_compatibility(map2) == {}

    @pytest.mark.parametrize("diff", [
        {'trajectory': traj.atom_slice([0, 1, 2, 3])},
        {'cutoff': 0.45},
        {'n_neighbors_ignored': 2},
        {'query': [1, 2, 3, 4]},
        {'haystack': [1, 2, 3, 4]}
    ])
    def test_check_compatibility_assertion_error(self, diff):
        params = {'trajectory': traj[0:2],
                  'cutoff': 0.075,
                  'n_neighbors_ignored': 0}
        params.update(diff)
        map2 = ContactFrequency(**params)
        with pytest.raises(AssertionError):
            self.map._check_compatibility(map2)

    def test_check_compatibility_runtime_error(self):
        map2 = ContactFrequency(trajectory=traj,
                                cutoff=0.45,
                                n_neighbors_ignored=2)
        with pytest.raises(RuntimeError):
            self.map._check_compatibility(map2, err=RuntimeError)

    @pytest.mark.parametrize("intermediate", ["dict", "json"])
    def test_serialization_cycle(self, intermediate):
        serializer, deserializer = {
            'json': (self.map.to_json, ContactFrequency.from_json),
            'dict': (self.map.to_dict, ContactFrequency.from_dict)
        }[intermediate]

        serialized = serializer()
        reloaded = deserializer(serialized)
        _contact_object_compare(self.map, reloaded)
        assert self.map == reloaded

    def test_hash(self):
        map2 = ContactFrequency(trajectory=traj,
                                cutoff=0.075,
                                n_neighbors_ignored=0)
        map3 = ContactFrequency(trajectory=traj[:2],
                                cutoff=0.075,
                                n_neighbors_ignored=0)

        assert hash(self.map) == hash(map2)
        assert hash(self.map) != hash(map3)

    def test_saving(self):
        m = self.map
        m.save_to_file(test_file)
        m2 = ContactFrequency.from_file(test_file)
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

    def test_add_contact_frequency(self):
        # self.map has all 5 frames
        # we can figure out what the [0:4] would look like
        start = ContactFrequency(trajectory=traj[:4],
                                 cutoff=0.075,
                                 n_neighbors_ignored=0)
        add_in = ContactFrequency(trajectory=traj[4:],
                                  cutoff=0.075,
                                  n_neighbors_ignored=0)

        start.add_contact_frequency(add_in)

        assert start.atom_contacts.counter == \
            self.map.atom_contacts.counter

        assert start.residue_contacts.counter == \
            self.map.residue_contacts.counter

    def test_subtract_contact_frequency(self):
        first_four = ContactFrequency(trajectory=traj[:4],
                                      cutoff=0.075,
                                      n_neighbors_ignored=0)
        last_frame = ContactFrequency(trajectory=traj[4:],
                                      cutoff=0.075,
                                      n_neighbors_ignored=0)
        test_subject = ContactFrequency(trajectory=traj,
                                        cutoff=0.075,
                                        n_neighbors_ignored=0)

        test_subject.subtract_contact_frequency(first_four)

        assert test_subject.atom_contacts.counter == \
            last_frame.atom_contacts.counter

        assert test_subject.residue_contacts.counter == \
            last_frame.residue_contacts.counter

    @pytest.mark.parametrize("use_atom_slice", [True, False, None])
    def test_use_atom_slice(self, use_atom_slice):
        # Set class default before init
        class_default = ContactFrequency._class_use_atom_slice
        ContactFrequency._class_use_atom_slice = use_atom_slice
        mapq = ContactFrequency(trajectory=traj, cutoff=0.075,
                                n_neighbors_ignored=0, query=self.atoms)

        maph = ContactFrequency(trajectory=traj, cutoff=0.075,
                                n_neighbors_ignored=0, haystack=self.atoms)

        mapb = ContactFrequency(trajectory=traj, cutoff=0.075,
                                n_neighbors_ignored=0, query=self.atoms,
                                haystack=self.atoms)

        maps = [mapq, maph, mapb]
        atoms = {mapq: list(range(10)),
                 maph: list(range(10)),
                 mapb: self.atoms}
        expected_atom_slice = {mapq: False,
                               maph: False,
                               mapb: True}
        # Test init
        for m in maps:
            self.map = m
            assert m.all_atoms == atoms[m]
            atom_list = [traj.topology.atom(i) for i in m.all_atoms]
            check_use_atom_slice(m, use_atom_slice, expected_atom_slice)
            sliced_traj = m.indexer.slice_trajectory(traj)
            if m.use_atom_slice:
                assert sliced_traj.topology.n_atoms == len(m.all_atoms)
            else:
                assert sliced_traj is traj

            # Test counters
            self.test_counters()
        # Reset class default as pytest does not re-import
        ContactFrequency._class_use_atom_slice = class_default


class TestContactDifference(object):
    def test_diff_traj_frame(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
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

    @pytest.mark.parametrize("intermediate", ["dict", "json"])
    def test_serialization_cycle(self, intermediate):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        diff = ttraj - frame

        serializer, deserializer = {
            'json': (diff.to_json, ContactDifference.from_json),
            'dict': (diff.to_dict, ContactDifference.from_dict)
        }[intermediate]

        serialized = serializer()
        reloaded = deserializer(serialized)
        _contact_object_compare(diff, reloaded)
        assert diff == reloaded

    def test_diff_frame_frame(self):
        m0 = ContactFrequency(traj[0], cutoff=0.075, n_neighbors_ignored=0)
        m4 = ContactFrequency(traj[4], cutoff=0.075, n_neighbors_ignored=0)
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

    def test_contacts_dict(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        _check_contacts_dict_names(ttraj - frame)

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
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        diff = ttraj - frame

        diff.save_to_file(test_file)
        reloaded = ContactDifference.from_file(test_file)
        assert diff.atom_contacts.counter == reloaded.atom_contacts.counter
        os.remove(test_file)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="Missing matplotlib")
    def test_plot(self):
        # smoke test; checks that we cover negative counts in plotting
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        diff = ttraj - frame
        diff.residue_contacts.plot()

    @pytest.mark.parametrize("attr", [('query',[0]), ('haystack',[1]),
                                      ('cutoff', 0.07),
                                      ('n_neighbors_ignored',1)])

    def test_non_important_attributes(self, attr):
        kwargs = {'cutoff':0.075,
                  'n_neighbors_ignored':0}
        kwargs[attr[0]] = attr[1]
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], **kwargs)
        diff = ttraj - frame
        # Make sure the attributes are dead
        if attr[0] in {'query','haystack'}:
            assert getattr(diff, attr[0]) == attr[1]
        else:
            assert getattr(diff, attr[0]) is None
        # Make sure we can still do the maps
        assert diff.atom_contacts is not None
        assert diff.residue_contacts is not None

    def test_non_important_atoms(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to add an atom, which should not be part of the maps
        res = list(frame.topology.residues)[-1]
        elem = md.element.get_by_symbol("H")
        _ = frame.topology.add_atom(name='test',
                                    element=elem,
                                    residue=res,
                                    serial=42)
        diff = ttraj - frame
        # Make sure the topology does not fail here
        assert diff.atom_contacts is not None
        assert diff.residue_contacts is not None

    def test_non_important_residues(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to add a residue,
        # which should not be part of the maps
        chain = list(frame.topology.chains)[-1]
        _ = frame.topology.add_residue(name='test',
                                       chain=chain)
        diff = ttraj - frame
        # Make sure the topology does not fail here
        assert diff.atom_contacts is not None
        assert diff.residue_contacts is not None

    def test_truncated_diffs_residues(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        traj_trunc = traj.atom_slice(range(6))
        ttraj_trunc = ContactFrequency(traj_trunc[0:4], cutoff=0.075,
                                       n_neighbors_ignored=0)
        frame_trunc = ContactFrequency(traj_trunc[4], cutoff=0.075,
                                       n_neighbors_ignored=0)
        # The diff should only be made for atoms and residues in both maps
        diff1 = ttraj - frame_trunc
        diff2 = ttraj_trunc - frame_trunc
        # Make sure the topology does not fail here
        assert diff1.atom_contacts is not None
        assert diff1.residue_contacts is not None

        # Make sure the diffs are equal
        assert diff1.atom_contacts.counter == diff2.atom_contacts.counter
        assert diff1.residue_contacts.counter == diff2.residue_contacts.counter

    def test_residue_rename_gives_different_atoms(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to rename a residue,
        frame.topology.residue(0).name = 'test'
        # Make sure the topology does not fail here
        diff = ttraj - frame

        assert diff.atom_contacts is not None
        assert diff.residue_contacts is not None
        assert diff.topology.residue(0).name == "XXX/test"

    def test_broken_atoms_altered(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to change an element,
        elem = md.element.get_by_symbol("H")
        frame.topology.atom(0).element = elem
        # Make sure the topology does not fail here
        with pytest.raises(RuntimeError) as e:
            diff = ttraj - frame
        assert "AtomMismatch" in str(e.value)

    def test_broken_atoms_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to delete an atom,
        frame.topology.delete_atom_by_index(3)
        with pytest.raises(RuntimeError) as e:
            diff = ttraj - frame
        assert "AtomMismatch" in str(e.value)

    def test_broken_residues(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Somehow only altering residues per atom is not an unequal
        # topology in mdtraj, so we just add one for mdtraj to trigger
        chain = frame.topology.chain(0)
        _ = frame.topology.add_residue(name='test',
                                       chain=chain)
        frame.topology.residue(0).resSeq = "test"

        with pytest.raises(RuntimeError) as e:
            diff = ttraj - frame
        assert "ResidueMismatch" in str(e.value)

    def test_broken_residues_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Grab atom info to add 'equal' atoms at the end
        element8 = frame.topology.atom(8).element
        element9 = frame.topology.atom(9).element

        name8 = frame.topology.atom(8).name
        name9 = frame.topology.atom(9).name

        serial8 = frame.topology.atom(8).serial
        serial9 = frame.topology.atom(9).serial


        # Remove residue from the internals of the topology
        frame._topology = frame.topology.subset(range(8))
        assert frame.topology.n_residues == 4
        # Add the original atoms again
        res = frame.topology.atom(7).residue
        frame.topology.add_atom(name8, element8, res, serial8)
        frame.topology.add_atom(name9, element9, res, serial9)


        with pytest.raises(RuntimeError) as e:
            diff = ttraj - frame
        assert "ResidueMismatch" in str(e.value)

    def test_broken_everything_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # subset the topology
        frame._topology = frame.topology.subset(range(8))
        assert frame.topology.n_residues == 4


        with pytest.raises(RuntimeError) as e:
            diff = ttraj - frame
        assert "OverrideTopology" in str(e.value)


class TestAtomMismatchedContactDifference(object):
    def test_disabled_functions(self):
        disabled_funcs = ['atom_contacts',
                          'most_common_atoms_for_contact',
                          'most_common_atoms_for_residue',
                          'haystack_residues',
                          'query_residues']

        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        diff = AtomMismatchedContactDifference(ttraj, frame)
        for f in disabled_funcs:
            with pytest.raises(RuntimeError) as e:
                func = getattr(diff, f)
                # This should only trigger if f is not a property
                # (otherwise the previous line should raise)
                func()
            # test that 'atoms' is in the error
            assert 'atom' in str(e.value)
        # Make sure residue contacts are still accessible
        _ = diff.residue_contacts

    def test_broken_atoms_altered(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to change an element,
        elem = md.element.get_by_symbol("H")
        frame.topology.atom(0).element = elem

        # Make sure this now works
        diff = AtomMismatchedContactDifference(ttraj, frame)
        assert diff.residue_contacts is not None

    def test_broken_atoms_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to delete an atom; this needs to be the last atom
        # otherwise we trigger on a short-circuit
        frame.topology.delete_atom_by_index(9)

        # Also need to make sure the name of the residue is different to not
        # trigger a missing residue
        frame.topology.residue(4).name = 'test'

        # Make sure this now works
        diff = AtomMismatchedContactDifference(ttraj, frame)
        assert diff.residue_contacts is not None

class TestResidueMismatchedContactDifference(object):
    def test_disabled_functions(self):
        disabled_funcs = ['residue_contacts',
                          '_residue_ignore_atom_idxs',
                          'most_common_atoms_for_contact',
                          'most_common_atoms_for_residue',
                          'haystack_residues',
                          'query_residues']

        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        diff = ResidueMismatchedContactDifference(ttraj, frame)
        for f in disabled_funcs:
            with pytest.raises(RuntimeError) as e:
                func = getattr(diff, f)
                # This should only trigger if f is not a property
                # (otherwise the previous line should raise)
                func()
            # test that 'atoms' is in the error
            assert 'residue' in str(e.value)
        # Make sure residue contacts are still accessible
        _ = diff.atom_contacts


    def test_broken_residues_altred(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Somehow only altering residues per atom is not an unequal
        # topology in mdtraj, so we just add one for mdtraj to trigger
        chain = frame.topology.chain(0)
        _ = frame.topology.add_residue(name='test',
                                       chain=chain)
        res = frame.topology.residue(0)

        frame.topology.atom(9).residue = res
        frame.topology.atom(8).residue = res

        assert frame.topology.atom(9).residue.index == 0
        assert frame.topology.atom(8).residue.index == 0
        assert ttraj.topology.atom(9).residue.index == 4
        assert ttraj.topology.atom(8).residue.index == 4


        diff = ResidueMismatchedContactDifference(ttraj, frame)
        assert diff.atom_contacts is not None


    def test_broken_residues_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Grab atom info to add 'equal' atoms at the end
        element8 = frame.topology.atom(8).element
        element9 = frame.topology.atom(9).element

        name8 = frame.topology.atom(8).name
        name9 = frame.topology.atom(9).name

        serial8 = frame.topology.atom(8).serial
        serial9 = frame.topology.atom(9).serial

        # subset the topology
        frame._topology = frame.topology.subset(range(8))
        assert frame.topology.n_residues == 4
        # Add the original atoms again
        res = frame.topology.atom(7).residue
        frame.topology.add_atom(name8, element8, res, serial8)
        frame.topology.add_atom(name9, element9, res, serial9)

        diff = ResidueMismatchedContactDifference(ttraj, frame)
        assert diff.atom_contacts is not None


class TestOverrideTopologyContactDifference(object):
    def test_broken_atoms_and_residues_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        # Now we are going to delete an atom; this needs to be the last atom
        # otherwise we trigger on a short-circuit
        frame.topology.delete_atom_by_index(9)

        # Make sure this now works
        diff = OverrideTopologyContactDifference(ttraj, frame, ttraj.topology)

        assert diff.residue_contacts is not None
        assert diff.atom_contacts is not None
        assert diff.topology == ttraj.topology
        assert diff.topology != frame.topology

    def test_still_broken_atoms_and_residues_missing(self):
        ttraj = ContactFrequency(traj[0:4], cutoff=0.075,
                                 n_neighbors_ignored=0)
        frame = ContactFrequency(traj[4], cutoff=0.075,
                                 n_neighbors_ignored=0)

        frame.topology.delete_atom_by_index(9)

        # Make sure still break
        with pytest.raises(RuntimeError) as e:
            diff = OverrideTopologyContactDifference(ttraj, frame,
                                                     frame.topology)
        assert 'AtomMismatched' in str(e.value)
