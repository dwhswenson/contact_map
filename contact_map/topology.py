import mdtraj as md


def check_atoms_ok(top0, top1, atoms):
    """Check if two topologies are equal on an atom level"""
    genatom = (top0.atom(i) == top1.atom(i) for i in atoms)
    # This needs try except in case one topology does not have the atom index
    try:
        all_atoms_ok = all(genatom)
    except IndexError:
        # If a given topology does not containthe required indices.
        all_atoms_ok = False
    return all_atoms_ok


def check_residues_ok(top0, top1, atoms, out_topology=None):
    """Check if the residues in two topologies are equal.

    If an out_topology is given, and the residues only differ in name, that
    residue name will be updated inplace in the out_topology
    """
    res_idx = _get_residue_indices(top0, top1, atoms)
    all_res_ok = (bool(len(res_idx)))  # True if is bigger than 0
    all_res_ok &= not _count_mismatching_names(top0, top1, res_idx,
                                               out_topology)

    return all_res_ok


def _get_residue_indices(top0, top1, atoms):
    """Get the residue indices or an empty list if not equal."""
    out_idx = []
    try:
        res_idx0 = set([top0.atom(i).residue.index for i in atoms])
        res_idx1 = set([top1.atom(i).residue.index for i in atoms])
    except IndexError:
        return out_idx

    # Check if the involved indices are equal
    if res_idx0 == res_idx1:
        out_idx = res_idx0
    return out_idx


def _count_mismatching_names(top0, top1, residx, out_topology=None):
    """Check for mismatching names.

    This will return truthy value if found and not fixable.
    """
    # Check if the names are different
    mismatched_idx = []
    for idx in residx:
        name0 = top0.residue(idx).name
        name1 = top1.residue(idx).name
        if name0 != name1:
            mismatched_idx.append((idx, name0, name1))
    if out_topology:
        return _fix_topology(mismatched_idx, out_topology)
    return len(mismatched_idx)


def _fix_topology(mismatched_idx, out_topology):
    """Try to fix the topology, will return true if this errors out"""
    for idx, name0, name1 in mismatched_idx:
        try:
            out_topology.residue(idx).name = "/".join([name0, name1])
        except IndexError:
            # If out topology is not complete
            return True
    return False


def check_topologies(map0, map1, override_topology):
    """Check if the topologies of two contact maps are ok or can be fixed"""
    # Grab the two topologies
    top0 = map0.topology
    top1 = map1.topology

    # Make a custom topology
    if isinstance(override_topology, md.Topology):
        # User provided topology
        topology = override_topology
        top0 = topology
        top1 = topology
    elif top0.n_atoms >= top1.n_atoms:
        # assume the topology of the bigger system contains the smaller one
        topology = top0.copy()
    else:
        topology = top1.copy()

    # Figure out the overlapping atoms
    all_atoms0 = map0.query | map0.haystack
    all_atoms1 = map1.query | map1.haystack

    # Should this be an overlap or a join?
    overlap_atoms = all_atoms0 | all_atoms1
    all_atoms_ok = check_atoms_ok(top0, top1, overlap_atoms)
    all_res_ok = check_residues_ok(top0, top1, overlap_atoms, topology)
    if not all_res_ok and not all_atoms_ok:
        topology = md.Topology()

    return all_atoms_ok, all_res_ok, topology
