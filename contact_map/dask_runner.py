"""
Implementation of ContactFrequency parallelization using dask.distributed
"""

from . import frequency_task
from .contact_map import ContactFrequency, ContactObject
import mdtraj as md


def dask_run(trajectory, client, run_info):
    """
    Runs dask version of ContactFrequency. Note that this API on this will
    definitely change before the release.

    Parameters
    ----------
    trajectory : mdtraj.trajectory
    client : dask.distributed.Client
        path to dask scheduler file
    run_info : dict
        keys are 'trajectory_file' (trajectory filename), 'load_kwargs'
        (additional kwargs passed to md.load), and 'parameters' (dict of
        kwargs for the ContactFrequency object)

    Returns
    -------
    :class:`.ContactFrequency` :
        total contact frequency for the trajectory
    """
    slices = frequency_task.default_slices(n_total=len(trajectory),
                                           n_workers=len(client.ncores()))

    subtrajs = client.map(frequency_task.load_trajectory_task, slices,
                          file_name=run_info['trajectory_file'],
                          **run_info['load_kwargs'])
    maps = client.map(frequency_task.map_task, subtrajs,
                      parameters=run_info['parameters'])
    freq = client.submit(frequency_task.reduce_all_results, maps)

    return freq.result()


class DaskContactFrequency(ContactFrequency):
    """Dask-based parallelization of contact frequency.

    The contact frequency is the fraction of a trajectory that a contact is
    made. See :class:`.ContactFrequency` for details. This implementation
    parallelizes the contact frequency calculation using
    ``dask.distributed``, which must be installed separately to use this
    object.

    Notes
    -----

    The interface for this object closely mimics that of the
    :class:`.ContactFrequency` object, with the addition requiring the
    ``dask.distributed.Client`` as input. However, there is one important
    difference.  Whereas :class:`.ContactFrequency` takes an
    ``mdtraj.Trajectory`` object as input, :class:`.DaskContactFrequency`
    takes a file name, plus any extra kwargs that MDTraj needs to load the
    file.

    Parameters
    ----------
    client : dask.distributed.Client
        Client object connected to the dask network.
    filename : str
        Name of the file where the trajectory is located. File must be
        accessible by all workers in the dask network.
    query : list of int
        Indices of the atoms to be included as query. Default ``None``
        means all atoms.
    haystack : list of int
        Indices of the atoms to be included as haystack. Default ``None``
        means all atoms.
    cutoff : float
        Cutoff distance for contacts, in nanometers. Default 0.45.
    n_neighbors_ignored : int
        Number of neighboring residues (in the same chain) to ignore.
        Default 2.
    """
    def __init__(self, client, filename, query=None, haystack=None,
                 cutoff=0.45, n_neighbors_ignored=2, **kwargs):
        self.client = client
        self.filename = filename
        trajectory = md.load(filename, **kwargs)

        self.kwargs = kwargs

        super(DaskContactFrequency, self).__init__(
            trajectory, query, haystack, cutoff, n_neighbors_ignored,
        )

    def _build_contact_map(self, trajectory):
        freq = dask_run(trajectory, self.client, self.run_info)
        self._frames = freq.n_frames
        return (freq._atom_contacts, freq._residue_contacts)

    @property
    def parameters(self):
        return {'query': self.query,
                'haystack': self.haystack,
                'cutoff': self.cutoff,
                'n_neighbors_ignored': self.n_neighbors_ignored}

    @property
    def run_info(self):
        return {'parameters': self.parameters,
                'trajectory_file': self.filename,
                'load_kwargs': self.kwargs}
