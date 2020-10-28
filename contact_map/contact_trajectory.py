from collections import abc, Counter

from .contact_map import ContactFrequency, ContactObject
import json

class ContactTrajectory(ContactObject, abc.Sequence):
    """Track all the contacts over a trajectory, frame-by-frame.

    Internally, this has a single-frame :class:`.ContactFrequency` for each
    frame of the trajectory.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        the trajectory to calculate contacts for
    query : list of int
        Indices of the atoms to be included as query. Default ``None``
        means all heavy, non-water atoms.
    haystack : list of int
        Indices of the atoms to be included as haystack. Default ``None``
        means all heavy, non-water atoms.
    cutoff : float
        Cutoff distance for contacts, in nanometers. Default 0.45.
    n_neighbors_ignored : int
        Number of neighboring residues (in the same chain) to ignore.
        Default 2.
    """
    _class_use_atom_slice = None
    def __init__(self, trajectory, query=None, haystack=None, cutoff=0.45,
                 n_neighbors_ignored=2):
        super(ContactTrajectory, self).__init__(trajectory.topology, query,
                                                haystack, cutoff,
                                                n_neighbors_ignored)
        contacts = self._build_contacts(trajectory)
        self._contact_maps = [
            ContactFrequency.from_contacts(
                topology=self.topology,
                query=self.query,
                haystack=self.haystack,
                cutoff=self.cutoff,
                n_neighbors_ignored=self.n_neighbors_ignored,
                atom_contacts=atom_contacts,
                residue_contacts=residue_contacts,
                n_frames=1,
                indexer=self.indexer
            )
            for atom_contacts, residue_contacts in zip(*contacts)
        ]

    def __getitem__(self, num):
        return self._contact_maps[num]

    def __len__(self):
        return len(self._contact_maps)

    def __hash__(self):
        return hash((super(ContactTrajectory, self).__hash__(),
                     tuple([frozenset(frame.counter.items())
                            for frame in self.atom_contacts]),
                     tuple([frozenset(frame.counter.items())
                            for frame in self.residue_contacts])))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @classmethod
    def from_contacts(cls, atom_contacts, residue_contacts, topology,
                      query=None, haystack=None, cutoff=0.45,
                      n_neighbors_ignored=2):
        contact_maps = [
            ContactFrequency.from_contacts(
                atom_cs,
                res_cs,
                n_frames=1,
                topology=topology,
                query=query,
                haystack=haystack,
                cutoff=cutoff,
                n_neighbors_ignored=n_neighbors_ignored
            )
            for atom_cs, res_cs in zip(atom_contacts, residue_contacts)
        ]
        return cls.from_contact_maps(contact_maps)

    def _build_contacts(self, trajectory):
        # atom_contacts, residue_contacts = self._empty_contacts()
        atom_contacts = []
        residue_contacts = []

        residue_ignore_atom_idxs = self._residue_ignore_atom_idxs
        residue_query_atom_idxs = self.indexer.residue_query_atom_idxs
        used_trajectory = self.indexer.slice_trajectory(trajectory)

        # range(len(trajectory)) avoids recopying topology, as would occur
        # in `for frame in trajectory`
        for frame_num in range(len(trajectory)):
            frame_contacts = self._contact_map(used_trajectory, frame_num,
                                               residue_query_atom_idxs,
                                               residue_ignore_atom_idxs)
            frame_atom_contacts, frame_residue_contacts = frame_contacts
            frame_atom_contacts = \
                    self.indexer.convert_atom_contacts(frame_atom_contacts)
            # TODO unify contact building with something like this?
            # atom_contacts, residue_contact = self._update_contacts(...)
            atom_contacts.append(frame_atom_contacts)
            residue_contacts.append(frame_residue_contacts)
        return atom_contacts, residue_contacts

    def contact_frequency(self):
        """Create a :class:`.ContactFrequency` from this contact trajectory
        """
        freq = ContactFrequency.from_contacts(
            atom_contacts=Counter(),
            residue_contacts=Counter(),
            n_frames=0,
            topology=self.topology,
            query=self.query,
            haystack=self.haystack,
            cutoff=self.cutoff,
            n_neighbors_ignored=self.n_neighbors_ignored
        )
        for cmap in self._contact_maps:
            # TODO: skipping compatibility checks would help performance; we
            # know that everything in here *should* be compatible
            freq.add_contact_frequency(cmap)

        return freq

    def to_dict(self):
        return {
            'contact_maps': [cmap.to_dict() for cmap in self._contact_maps]
        }

    @classmethod
    def from_dict(cls, dct):
        contact_maps = [ContactFrequency.from_dict(cmap)
                        for cmap in dct['contact_maps']]
        obj = cls.from_contact_maps(contact_maps)
        return obj

    @property
    def atom_contacts(self):
        return [cmap.atom_contacts for cmap in self._contact_maps]

    @property
    def residue_contacts(self):
        return [cmap.residue_contacts for cmap in self._contact_maps]

    @classmethod
    def from_contact_maps(cls, maps):
        obj = cls.__new__(cls)
        super(cls, obj).__init__(maps[0].topology, maps[0].query,
                                 maps[0].haystack, maps[0].cutoff,
                                 maps[0].n_neighbors_ignored)

        for cmap in maps:
            obj._check_compatibility(cmap)

        obj._contact_maps = maps
        return obj

    @classmethod
    def join(cls, others):
        """Concatenate ContactTrajectory instances

        Parameters
        ----------
        others : List[:class:.ContactTrajectory]
            contact trajectories to concatenate

        Returns
        -------
        :class:`.ContactTrajectory` :
            concatenated contact trajectory
        """
        contact_maps = sum([o._contact_maps for o in others], [])
        return cls.from_contact_maps(contact_maps)

    def rolling_frequency(self, window_size=1, step=1):
        """:class:`.RollingContactFrequency` iterator for this trajectory

        Parameters
        ----------
        window_size : int
            the number of frames in the window
        step : int
            the number of frames between successive starting points of the
            window (like the ``step`` parameter in a Python slice object)

        Returns
        -------
        :class:`.RollingContactFrequency` :
            windowed iterator for this trajectory
        """
        return RollingContactFrequency(self, width=window_size, step=step)


class MutableContactTrajectory(ContactTrajectory, abc.MutableSequence):
    """Mutable version of :class:`.ContactTrajectory`

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        the trajectory to calculate contacts for
    query : list of int
        Indices of the atoms to be included as query. Default ``None``
        means all heavy, non-water atoms.
    haystack : list of int
        Indices of the atoms to be included as haystack. Default ``None``
        means all heavy, non-water atoms.
    cutoff : float
        Cutoff distance for contacts, in nanometers. Default 0.45.
    n_neighbors_ignored : int
        Number of neighboring residues (in the same chain) to ignore.
        Default 2.

    """
    def __setitem__(self, key, value):
        self._contact_maps[key] = value

    def __delitem__(self, key):
        del self._contact_maps[key]

    def insert(self, key, value):
        self._contact_maps.insert(key, value)

    def __hash__(self):
        # mutable objects must have unique hashes
        return id(self)


class WindowedIterator(abc.Iterator):
    """
    Helper for windowed ("rolling average") iterators.

    The idea is that this is an easy and reusable code for getting windowed
    quantitiies such as needed for rolling averages. This iterator itself
    just returns sets of indices/slices to add/remove from whatever counter
    is being tracked. The idea is that it will be used inside of another
    iterator.


    Parameters
    ----------
    length : int
        the length of the list windowed over
    width : int
        the number of items in the window
    step : int
        the number of items skipped between successive windows (as with the
        ``step`` parameter in slices)
    slow_build : bool
        if True, the iterator builds up the window "step" objects at a time.
        Otherwise, the first value is the full width of the window.

    Attributes
    ----------
    min : int
        the index of the first object in the cached window
    max : int
        the index of the last object in the cached window (note that this is
        included in the window, unlike Python slices)
    """
    def __init__(self, length, width, step, slow_build):
        self.length = length
        self.width = width
        self.step = step
        self.slow_build = slow_build
        self.min = -1
        self.max = -1

    def _startup(self):
        to_sub = slice(0, 0)
        self.min = max(self.min, 0)
        if self.slow_build:
            to_add = slice(self.max + 1, self.max + self.step + 1)
            self.max += self.step
        else:
            self.max = self.width - 1
            to_add = slice(self.min, self.max + 1)
        return to_add, to_sub

    def _normal(self):
        self.min = max(0, self.min)
        new_max = self.max + self.step

        if not self.slow_build:
            new_max = max(new_max, self.width - 1)

        new_min = max(self.min, new_max - self.width + 1)

        to_sub = slice(self.min, new_min)
        to_add = slice(self.max + 1, new_max + 1)
        self.min = new_min
        self.max = new_max
        return to_add, to_sub

    def __next__(self):
        # if self.max + self.step < self.width:
            # to_add, to_sub = self._startup()
        if self.max + self.step < self.length:
            to_add, to_sub = self._normal()
        else:
            raise StopIteration

        return to_add, to_sub


class RollingContactFrequency(abc.Iterator):
    """Iterator for "rolling-average" contact frequencies over a trajectory

    Parameters
    ----------
    contact_trajectory : :class:`.ContactTrajectory`
        input trajectory
    width : int
        the number of frames in the window
    step : int
        the number of frames between successive starting points of the
        window (like the ``step`` parameter in a Python slice object)
    """

    _slow_build_iter = False

    def __init__(self, contact_trajectory, width=1, step=1):
        self.trajectory = contact_trajectory
        self.width = width
        self.step = step
        self.slow_build_iter = self._slow_build_iter
        self._window_iter = None
        self._contact_map = None

    def __iter__(self):
        self._window_iter = WindowedIterator(length=len(self.trajectory),
                                             width=self.width,
                                             step=self.step,
                                             slow_build=self.slow_build_iter)
        self._contact_map = ContactFrequency.from_contacts(
            Counter(), Counter(),
            topology=self.trajectory.topology,
            query=self.trajectory.query,
            haystack=self.trajectory.haystack,
            cutoff=self.trajectory.cutoff,
            n_neighbors_ignored=self.trajectory.n_neighbors_ignored,
            n_frames=0
        )
        return self

    def __next__(self):
        to_add, to_sub = next(self._window_iter)
        for frame in self.trajectory[to_add]:
            self._contact_map.add_contact_frequency(frame)
        for frame in self.trajectory[to_sub]:
            self._contact_map.subtract_contact_frequency(frame)

        # need to make a copy in case the user does list(rolling_freq),
        # otherwise they get copies of only the last version!
        cmap = self._contact_map
        map_copy = ContactFrequency.from_contacts(
            cmap._atom_contacts.copy(),
            cmap._residue_contacts.copy(),
            topology=cmap.topology,
            query=cmap.query,
            haystack=cmap.haystack,
            cutoff=cmap.cutoff,
            n_neighbors_ignored=cmap.n_neighbors_ignored,
            n_frames=cmap.n_frames
        )
        return map_copy
