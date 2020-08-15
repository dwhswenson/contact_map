import collections

class AtomSlicedIndexer(object):
    def __init__(self, topology, real_query, real_haystack, all_atoms):
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

        real_atom_idx_to_residue_idx = {atom.index: atom.residue.index
                                        for atom in topology.atoms}
        self.atom_idx_to_residue_idx = {
            sliced_idx: real_atom_idx_to_residue_idx[real_idx]
            for sliced_idx, real_idx in enumerate(all_atoms)
        }

    def convert_atom_contacts(self, atom_contacts):
        result =  {frozenset(map(self.real_idx.__getitem__, pair)): value
                   for pair, value in atom_contacts.items()}
        return collections.Counter(result)


class IdentityIndexer(object):
    def __init__(self, topology, real_query, real_haystack, all_atoms):
        identity_mapping = {a: a for a in range(topology.n_atoms)}
        self.sliced_idx = identity_mapping
        self.real_idx = identity_mapping
        self.query = set(real_query)
        self.haystack = set(real_haystack)
        self.atom_idx_to_residue_idx = {atom.index: atom.residue.index
                                        for atom in topology.atoms}

    def convert_atom_contacts(self, atom_contacts):
        return atom_contacts
