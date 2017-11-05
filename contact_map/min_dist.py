import collections
import itertools
import mdtraj as md

class NearestAtoms(object):
    """
    Identify nearest atoms (within a cutoff) to an atom.

    This was primarily written to quickly look for atoms that are nearly
    overlapping, but should be extendable to have other uses.

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


    Attributes
    ----------
    nearest : dict
        dictionary mapping atom index to the atom index of the nearest atom
        to this one
    nearest_distance : dict
        dictionary mapping atom index to the distance to the nearest atom
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

    @staticmethod
    def _calculate_nearest(trajectory, cutoff, frame_number, excluded):
        """
        Calculate the nearest atoms from the input data.

        Useful in alterative constructors. See class docs for parameters.
        """
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
        """Regularize the input value of ``excluded``.

        ``excluded`` input can be:
            * dict of {atom_index: [excluded_atom_indices}. This is the
              desired format, but a pain for users to create in many common
              cases.
            * None. This is the default; in this case, this function returns
              a dict in the desired format allowing all atoms not in the
              same residue as the atom_index.
            * empty dict ``{}``. In this case, this function returns a dict
              in the desired format where no atoms are excluded (absolute
              nearest atoms)
        """
        if excluded == {}:
            excluded = {idx: [idx] for idx in range(trajectory.n_atoms)}
        if excluded is None:
            top = trajectory.topology
            excluded = {
                idx: [a.index for a in top.atom(idx).residue.atoms]
                for idx in range(trajectory.n_atoms)
            }
        return excluded

    @property
    def sorted_distances(self):
        """
        list :
            3-tuple (atom_index, nearest_atom_index, nearest_distance) for
            each atom, sorted by distance.
        """
        listed = [(atom, self.nearest[atom], dist)
                  for (atom, dist) in list(self.nearest_distance.items())]
        return list(sorted(listed, key=lambda tup: tup[2]))


class MinimumDistanceCounter(object):
    """Count how often each atom pair is the minimum distance.

    Parameters
    ----------
    trajectory : :class:`mdtraj.Trajectory`
        trajectory to be analyzed
    query : list
        list of the (integer) atom indices to use as the query
    haystack : list
        list of the (integer) atom indices to use as the haystack


    Attributes
    ----------
    topology : :class:`mdtraj.Topology`
        the topology object associated with the class
    atom_pairs : list
        list of 2-tuples representing atom index pairs to use when looking
        for the minimum distance
    minimum_distances : list
        the minimum distance between query group and haystack group at each
        frame of the trajectory
    """
    # count how many times each atom pair has minimum distance
    def __init__(self, trajectory, query, haystack):
        # TODO: better approach is to make the initialization based on
        # atom_pairs
        self.topology = trajectory.topology
        self.atom_pairs = list(itertools.product(query, haystack))
        self.minimum_distances, self._min_pairs = \
                self._compute_from_atom_pairs(trajectory, self.atom_pairs)

    @staticmethod
    def _compute_from_atom_pairs(trajectory, atom_pairs):
        """Compute minimum distances/atom index pairs from atom_pairs.

        Useful for alternative constructors.

        Parameters
        ----------
        trajectory : :class:`mdtraj.Trajectory`
            trajectory to be analyzed
        atom_pairs : list
            list of 2-tuples representing atom index pairs to use when
            looking for the minimum distance

        Returns
        -------
        minimum_distances : list
            the minimum distance between query group and haystack group at
            each frame of the trajectory
        min_pairs : list of 2-tuple
            the atom indices for the pair of atoms corresponding to the
            reported minimum distance each frame of the trajectory
        """
        distances = md.compute_distances(trajectory, atom_pairs)
        min_pairs = distances.argmin(axis=1)
        minimum_distances = distances.min(axis=1)
        return minimum_distances, min_pairs

    def _remap(self, pair_number):
        """Remap a pair of atom indices to the Atom objects"""
        pair = self.atom_pairs[pair_number]
        return (self.topology.atom(pair[0]), self.topology.atom(pair[1]))

    @property
    def atom_history(self):
        """
        list of 2-tuples :
            list of atom pairs when represent the minimum distance at each
            frame of the trajectory
        """
        return [self._remap(k) for k in self._min_pairs]

    @property
    def atom_count(self):
        """
        :class:`collections.Counter` :
            map from atom pair to the number of times that pair is the
            minimum distance
        """
        return collections.Counter(self.atom_history)

    @property
    def residue_history(self):
        """
        list of 2-tuples :
            list of residue pairs when represent the minimum distance at
            each frame of the trajectory
        """
        return [(a[0].residue, a[1].residue) for a in self.atom_history]

    @property
    def residue_count(self):
        """
        :class:`collections.Counter` :
            map from residue pair to the number of times that pair is the
            minimum distance
        """
        return collections.Counter(self.residue_history)
