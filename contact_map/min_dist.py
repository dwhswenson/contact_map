import collections
import itertools
import mdtraj as md

class NearestAtoms(object):
    """
    Identify nearest atoms (within a cutoff) to an atom.

    Parameters
    ----------
    trajectory : :class:`mdtraj.Trajectory`
        trajectory to be analyzed
    cutoff : float
        cutoff distance (in nm)
    frame_number : int
        frame number within the trajectory (counting from 0), default 0
    excluded : dict
        a dict of {atom_index: [excluded_atom_indices]}, where the excluded
        atom indices are atoms that should not be counted when considering
        the atom for the key atom_index. Default is ``None``, which ignores
        all atoms in the same residue. Passing an empty dict, ``{}``, will
        result in all atom pairs being considered
    """
    # TODO: this can probably be refactored to match the behavior of the
    # mindist object; can't be fully removed because this will be a more
    # expensive calc
    def __init__(self, trajectory, cutoff, frame_number=0, excluded=None):
        self.cutoff = cutoff
        self.frame_number = frame_number
        self.excluded = self._parse_excluded(excluded, trajectory)
        self.nearest, self.nearest_distance = \
                self._calculate_nearest(trajectory, self.cutoff,
                                        self.frame_number, self.excluded)


    def _calculate_nearest(self, trajectory, cutoff, frame_number,
                           excluded):
        neighborlist = md.compute_neighborlist(trajectory, cutoff,
                                               frame_number)
        nearest = {}
        nearest_distance = {}
        for (atom, neighbors) in enumerate(neighborlist):
            allowed_neighbors = [n for n in neighbors
                                 if n not in excluded[atom]]
            pairs = list(itertools.product([atom], allowed_neighbors))
            if len(pairs) != 0:
                distances = md.compute_distances(trajectory,
                                                 pairs)[frame_number]

                nearest_tuple = \
                        sorted(list(zip(distances, allowed_neighbors)))[0]
                nearest[atom] = nearest_tuple[1]
                nearest_distance[atom] = nearest_tuple[0]
        return (nearest, nearest_distance)

    @staticmethod
    def _parse_excluded(excluded, trajectory):
        if excluded == {}:
            excluded = {idx: [] for idx in range(trajectory.n_atoms)}
        if excluded is None:
            top = trajectory.topology
            excluded = {
                idx: [a.index for a in top.atom(idx).residue.atoms]
                for idx in range(trajectory.n_atoms)
            }
        return excluded

    @property
    def sorted_distances(self):
        listed = [(atom, self.nearest[atom], dist)
                  for (atom, dist) in list(self.nearest_distance.items())]
        return sorted(listed, key=lambda tup: tup[2])


class MinimumDistanceCounter(object):
    """Count how often each atom pair is the minimum distance.
    """
    # count how many times each atom pair has minimum distance
    def __init__(self, trajectory, query, haystack):
        # TODO: better approach is to make the initialization based on
        # atom_pairs
        self.topology = trajectory.topology
        self.atom_pairs = list(itertools.product(query, haystack))
        self.minimum_distances, self._min_pairs = \
                self._compute_from_atom_pairs(trajectory, self.atom_pairs)

    @classmethod
    def from_contact_map(cls, trajectory, contact_map):
        """contact_map based constructor"""
        min_dist = cls.__new__()
        min_dist.topology = trajectory.topology
        min_dist.atom_pairs = [list(c) for c in contact_map.atom_contacts]
        min_dist.minimum_distances, min_dist._min_pairs = \
                min_dist._compute_from_atom_pairs(trajectory,
                                                  min_dist.atom_pairs)
        return min_dist

    @staticmethod
    def _compute_from_atom_pairs(trajectory, atom_pairs):
        distances = md.compute_distances(trajectory, atom_pairs)
        min_pairs = distances.argmin(axis=1)
        minimum_distances = distances.min(axis=1)
        return minimum_distances, min_pairs

    def _remap(self, pair_number):
        pair = self.atom_pairs[pair_number]
        return (self.topology.atom(pair[0]), self.topology.atom(pair[1]))

    @property
    def atom_history(self):
        return [self._remap(k) for k in self._min_pairs]

    @property
    def atom_count(self):
        return collections.Counter(self.atom_history)

    @property
    def residue_history(self):
        return [(a[0].residue, a[1].residue) for a in self.atom_history]

    @property
    def residue_count(self):
        return collections.Counter(self.residue_history)
