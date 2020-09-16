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
    try:
        res_idx0 = set([top0.atom(i).residue.index for i in atoms])
        res_idx1 = set([top1.atom(i).residue.index for i in atoms])
    except IndexError:
        return False

    all_res_ok = False
    # Check if the involved indices are equal
    if res_idx0 == res_idx1:
        all_res_ok = True
        for idx in res_idx0:
            name0 = top0.residue(idx).name
            name1 = top1.residue(idx).name
            if name0 != name1 and out_topology:
                try:
                    out_topology.residue(idx).name = "/".join([name0, name1])
                except IndexError:
                    # If out topology is not complete
                    all_res_ok = False
                    break
            elif name0 != name1:
                all_res_ok = False
                break
    return all_res_ok


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
    else:
        # Assume the topology of the bigger system contains the smaller one
        if top0.n_atoms >= top1.n_atoms:
            topology = top0.copy()
        else:
            topology = top1.copy()

    # Figure out the overlapping atoms
    all_atoms0 = map0.query | map0.haystack
    all_atoms1 = map1.query | map1.haystack

    overlap_atoms = all_atoms0 | all_atoms1
    all_atoms_ok = check_atoms_ok(top0, top1, overlap_atoms)
    all_res_ok = check_residues_ok(top0, top1, overlap_atoms, topology)
    if not all_res_ok and not all_atoms_ok:
        topology = md.Topology()
    return all_atoms_ok, all_res_ok, topology
