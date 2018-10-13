import numpy as np

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

# includes pytest
from .utils import *

from contact_map.set_dict import *
from .test_contact_map import traj

def make_key(obj_type, iter_type, idx_to_type, idx_pair):
    top = traj.topology
    idx_to_type_f = {
        'idx': lambda idx: idx,
        'obj': {'atom': top.atom,
                'res': top.residue}[obj_type]
    }[idx_to_type]
    iter_type_f = {'list': list,
                   'tuple': tuple,
                   'fset': frozenset}[iter_type]
    key = iter_type_f(idx_to_type_f(idx) for idx in idx_pair)
    return key


class TestFrozenSetDict(object):
    def setup(self):
        topology = traj.topology
        self.expected_dct = {
            frozenset([0, 1]): 10,
            frozenset([1, 2]): 5
        }
        self.atom_fsdict = FrozenSetDict({
            (topology.atom(0), topology.atom(1)): 10,
            (topology.atom(1), topology.atom(2)): 5
        })
        self.residue_fsdct = FrozenSetDict({
            (topology.residue(0), topology.residue(1)): 10,
            (topology.residue(1), topology.residue(2)): 5
        })

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_init(self, obj_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        assert obj.dct == self.expected_dct

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_len(self, obj_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        assert len(obj) == 2

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    def test_iter(self, obj_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        for k in obj:
            assert k in [frozenset([0,1]), frozenset([1,2])]

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    @pytest.mark.parametrize("iter_type", ['list', 'tuple', 'fset'])
    @pytest.mark.parametrize("idx_to_type", ['idx', 'obj'])
    def test_get(self, obj_type, iter_type, idx_to_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        key = make_key(obj_type, iter_type, idx_to_type, [0, 1])
        assert obj[key] == 10


    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    @pytest.mark.parametrize("iter_type", ['list', 'tuple', 'fset'])
    @pytest.mark.parametrize("idx_to_type", ['idx', 'obj'])
    def test_set(self, obj_type, iter_type, idx_to_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        key = make_key(obj_type, iter_type, idx_to_type, [1, 3])
        obj[key] = 20
        assert obj.dct[frozenset([1, 3])] == 20

    @pytest.mark.parametrize("obj_type", ['atom', 'res'])
    @pytest.mark.parametrize("iter_type", ['list', 'tuple', 'fset'])
    @pytest.mark.parametrize("idx_to_type", ['idx', 'obj'])
    def test_del(self, obj_type, iter_type, idx_to_type):
        obj = {'atom': self.atom_fsdict,
               'res': self.residue_fsdct}[obj_type]
        key = make_key(obj_type, iter_type, idx_to_type, [0, 1])
        del obj[key]
        assert len(obj) == 1
        assert list(obj.dct.keys()) == [frozenset([1,2])]


class TestFrozenSetCounter(object):
    pass
