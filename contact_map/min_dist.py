import collections
import itertools
import mdtraj as md

class NearestAtoms(object):
    def __init__(self, trajectory, cutoff, frame_number=0):
        # TODO: add support for a subset of all atoms with `atoms`
        self.cutoff = cutoff
        self.frame_number = frame_number
        neighborlist = md.compute_neighborlist(trajectory, self.cutoff,
                                               self.frame_number)
        frame = trajectory[frame_number]
        self.nearest = {}
        self.nearest_distance = {}
        for (atom, neighbors) in enumerate(neighborlist):
            pairs = itertools.product([atom], neighbors)
            distances = md.compute_distances(frame, pairs)[0]  # 0th frame
            nearest = sorted(zip(distances, neighbors))[0]
            self.nearest[atom] = nearest[1]
            self.nearest_distance[atom] = nearest[0]

    @property
    def sorted_distances(self):
        listed = [(atom, self.nearest[atom], dist)
                  for (atom, dist) in list(self.nearest_distance.items())]
        return sorted(listed, key=lambda tup: tup[2])

class MinimumDistanceCounter(object):
    # count how many times each atom pair has minimum distance
    def __init__(self, trajectory, query, haystack, cutoff=0.45):
        self.atom_pairs = list(itertools.product(query, haystack))
        distances = md.compute_distances(trajectory,
                                         atom_pairs=self.atom_pairs)
        self._min_pairs = distances.argmin(axis=1)
        self.minimum_distances = distances.min(axis=1)
        self.topology = trajectory.topology
        self.cutoff = cutoff

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
