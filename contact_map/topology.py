import mdtraj as md


def check_atoms_ok(top0, top1, atoms):
    """Check if two topologies are equal on an atom level"""
    genatom = (atoms_eq(top0.atom(i), top1.atom(i)) for i in atoms)
    # This needs try except in case one topology does not have the atom index
    try:
        all_atoms_ok = all(genatom)
    except IndexError:
        # If a given topology does not contain the required indices.
        all_atoms_ok = False
    return all_atoms_ok


def atoms_eq(one, other):
    """Check if atoms are equal except from resname and residue"""
    checks = [one.name.rsplit('-', 1)[-1] == other.name.rsplit('-', 1)[-1],
              one.element == other.element,
              one.index == other.index,
              one.serial == other.serial]
    return all(checks)


def residue_eq(one, other):
    """Check if residues are equal except from name and chain"""
    checks = [one.index == other.index,
              one.resSeq == other.resSeq,
              one.segment_id == other.segment_id]
    return all(checks)


def check_residues_ok(top0, top1, residues, out_topology=None):
    """Check if the residues in two topologies are equal.

    If an out_topology is given, and the residues only differ in name, that
    residue name will be updated inplace in the out_topology
    """
    res_idx = _get_residue_indices(top0, top1, residues)

    all_res_ok = (bool(len(res_idx)))  # True if is bigger than 0

    all_res_ok &= not _count_mismatching_residues(top0, top1, res_idx,
                                                  out_topology)
    return all_res_ok


def _get_residue_indices(top0, top1, residues):
    """Get the residue indices or an empty set if not able."""
    for top in (top0, top1):
        try:
            res = [top.residue(i).index for i in residues]
        except IndexError:
            return {}
    return res


def _count_mismatching_residues(top0, top1, residx, out_topology=None):
    """Check for mismatching names.

    This will return truthy value if found and not fixable.
    It also assumes all indices are present in both topologies
    """
    # Check if the names are different
    mismatched_idx = []
    mismatched_other = []
    for idx in residx:
        res0 = top0.residue(idx)
        res1 = top1.residue(idx)
        if not residue_eq(res0, res1):
            mismatched_other.append(idx)
        elif res0.name != res1.name:
            mismatched_idx.append((idx, res0.name, res1.name))

    if out_topology:
        _fix_residue_names(mismatched_idx, out_topology)
        mismatched_idx = []
    return len(mismatched_idx)+len(mismatched_other)


def _fix_residue_names(mismatched_idx, out_topology):
    """Fix the topology, assumes all indices are present"""
    for idx, name0, name1 in mismatched_idx:
        out_topology.residue(idx).name = "/".join([name0, name1])


def _get_default_topologies(top0, top1, override_topology):
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
    return top0, top1, topology


def check_topologies(map0, map1, override_topology):
    """Check if the topologies of two contact maps are ok or can be fixed"""
    # Grab the two topologies
    top0 = map0.topology
    top1 = map1.topology

    # Figure out the overlapping atoms
    all_atoms0 = set(map0._all_atoms)
    all_atoms1 = set(map1._all_atoms)

    # Figure out overlapping residues
    all_residues0 = set(map0._all_residues)
    all_residues1 = set(map1._all_residues)

    # This is intersect (for difference)
    overlap_atoms = all_atoms0.intersection(all_atoms1)
    overlap_residues = all_residues0.intersection(all_residues1)
    top0, top1, topology = _get_default_topologies(top0, top1,
                                                   override_topology)

    if override_topology:
        override_topology = topology

    all_atoms_ok = check_atoms_ok(top0, top1, overlap_atoms)
    all_res_ok = check_residues_ok(top0, top1, overlap_residues,
                                   override_topology)
    if not all_res_ok and not all_atoms_ok:
        topology = md.Topology()

    return all_atoms_ok, all_res_ok, topology
