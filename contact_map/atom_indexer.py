import collections
import numpy as np
import mdtraj as md

def _atom_slice(traj, indices):
    """Mock MDTraj.atom_slice without rebuilding topology"""
    xyz = np.array(traj.xyz[:, indices], order='C')
    topology = traj.topology.copy()
    if traj._have_unitcell:
        unitcell_lengths = traj._unitcell_lengths.copy()
        unitcell_angles = traj._unitcell_angles.copy()
    else:
        unitcell_lengths = None
        unitcell_angles = None
    time = traj._time.copy()

    # Hackish to make the smart slicing work
    topology._atoms = indices
    topology._numAtoms = len(indices)
    return md.Trajectory(xyz=xyz, topology=topology, time=time,
                         unitcell_lengths=unitcell_lengths,
                         unitcell_angles=unitcell_angles)

def residue_query_atom_idxs(sliced_query, atom_idx_to_residue_idx):
    residue_query_atom_idxs = collections.defaultdict(list)
    for sliced_idx in sliced_query:
        residue_idx = atom_idx_to_residue_idx[sliced_idx]
        residue_query_atom_idxs[residue_idx].append(sliced_idx)
    return residue_query_atom_idxs


class AtomSlicedIndexer(object):
    """Indexer when using atom slicing.
    """
    def __init__(self, topology, real_query, real_haystack, all_atoms):
        self.all_atoms = all_atoms
        self.sliced_idx = {
            real_idx : sliced_idx
            for sliced_idx, real_idx in enumerate(all_atoms)
        }
        self.real_idx = {
            sliced_idx: real_idx
            for real_idx, sliced_idx in self.sliced_idx.items()
        }
        self.query = set([self.sliced_idx[q] for q in real_query])
        self.haystack = set([self.sliced_idx[h] for h in real_haystack])

        # atom_idx_to_residue_idx
        self.real_atom_idx_to_residue_idx = {atom.index: atom.residue.index
                                             for atom in topology.atoms}
        self.atom_idx_to_residue_idx = {
            sliced_idx: self.real_atom_idx_to_residue_idx[real_idx]
            for sliced_idx, real_idx in enumerate(all_atoms)
        }
        self.residue_query_atom_idxs = residue_query_atom_idxs(
            self.query, self.atom_idx_to_residue_idx
        )

    def ignore_atom_idx(self, atoms, all_atoms_set):
        result = set(atom.index for atom in atoms)
        result &= all_atoms_set
        result = set(self.sliced_idx[a] for a in result)
        return result

    def convert_atom_contacts(self, atom_contacts):
        result =  {frozenset(map(self.real_idx.__getitem__, pair)): value
                   for pair, value in atom_contacts.items()}
        return collections.Counter(result)

    def slice_trajectory(self, trajectory):
        # Prevent (memory) expensive atom slicing if not needed.
        # This check is also needed here because ContactFrequency slices the
        # whole trajectory before calling this function.
        if len(self.all_atoms) < trajectory.topology.n_atoms:
            sliced = _atom_slice(trajectory, self.all_atoms)
        else:
            sliced = trajectory
        return sliced


class IdentityIndexer(object):
    """Indexer when not using atom slicing.
    """
    def __init__(self, topology, real_query, real_haystack, all_atoms):
        self.all_atoms = all_atoms
        self.topology = topology
        identity_mapping = {a: a for a in range(topology.n_atoms)}
        self.sliced_idx = identity_mapping
        self.real_idx = identity_mapping
        self.query = set(real_query)
        self.haystack = set(real_haystack)
        self.real_atom_idx_to_residue_idx = {atom.index: atom.residue.index
                                             for atom in topology.atoms}
        self.atom_idx_to_residue_idx = self.real_atom_idx_to_residue_idx
        self.residue_query_atom_idxs = residue_query_atom_idxs(
            self.query, self.atom_idx_to_residue_idx
        )

    def ignore_atom_idx(self, atoms, all_atoms_set):
        return set(atom.index for atom in atoms)

    def convert_atom_contacts(self, atom_contacts):
        return atom_contacts

    def slice_trajectory(self, trajectory):
        return trajectory
