from .topology import check_topologies


class ParameterFixer(object):
    """Helper class to carry variables around to fix parameters"""
    def __init__(self,
                 allow_mismatched_atoms=False,
                 allow_mismatched_residues=False,
                 override_topology=True):
        self.allow_mismatched_atoms = allow_mismatched_atoms
        self.allow_mismatched_residues = allow_mismatched_residues
        self.override_topology = override_topology

    def get_parameters(self, positive, negative):
        """Get the required parameters to initialise ContactDifference"""
        failed = positive._check_compatibility(negative, return_failed=True)
        return self._fix_parameters(positive, negative, failed)

    def _fix_parameters(self, positive, negative, failed):
        # First make the default output
        output = {'topology': positive.topology,
                  'query': positive.query,
                  'haystack': positive.haystack,
                  'cutoff': positive.cutoff,
                  'n_neighbors_ignored': positive.n_neighbors_ignored}

        for fail in failed:
            if fail in {'query', 'haystack'}:
                fixed = []
            elif fail in {'cutoff', 'n_neighbors_ignored'}:
                # We just set them to None
                fixed = None
            elif fail == 'topology':
                # This requires quite a bit of logic
                fixed = self._check_topology(positive, negative)
            output[fail] = fixed
        return tuple(output.values())

    def _check_topology(self, positive, negative):
        all_atoms_ok, all_res_ok, topology = check_topologies(
            map0=positive,
            map1=negative,
            override_topology=self.override_topology
        )
        if not all_atoms_ok and not all_res_ok:
            # We don\t know how to fix this, defer to the user
            self._disable_all_contacts()

        if not all_atoms_ok:
            # Atom mapping does not make sense at the moment, override func
            # TODO: Might be fixable if all_atoms are equal length
            self._disable_atom_contacts()

        if not all_res_ok:
            # Can't be fixed for now
            # TODO: Can be fixed if the number of residues is equal
            # Or one is a subset of the other
            self._disable_residue_contacts()

        return topology

    def _disable_all_contacts(self):
        msg = (
            "The two different contact maps had atoms and residues that were "
            " not equal between the two topologies."
            " If you want to compare these, use "
            "`diff = OverrideTopologyContactDifference(map1, map2, topology)`"
            " with a mdtraj.Topology that contains all atoms and residues."
        )
        if not (self.allow_mismatched_atoms and
                self.allow_mismatched_residues):
            raise RuntimeError(msg)

    def _disable_atom_contacts(self):
        msg = (
            "The two different contact maps had atoms that were not equal "
            "between the two topologies. If you want to compare these "
            "use `diff = AtomMismatchedContactDifference(map1, map2)`.\n"
            "Alternatively, use "
            "`diff = OverrideTopologyContactDifference(map1, map2, topology)`"
            " with a mdtraj.Topology."

        )
        if not self.allow_mismatched_atoms:
            raise RuntimeError(msg)

    def _disable_residue_contacts(self):
        msg = (
            "The two different contact maps had residues that were not equal "
            "between the two topologies. If you want to compare these "
            "use `diff = ResidueMismatchedContactDifference(map1, map2)`.\n"
            "Alternatively, use "
            "`diff = OverrideTopologyContactDifference(map1, map2, topology)`"
            " with a mdtraj.Topology."
        )
        if not self.allow_mismatched_residues:
            raise RuntimeError(msg)
