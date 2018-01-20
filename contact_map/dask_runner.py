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
    def __init__(self, client, filename, query=None, haystack=None,
                 cutoff=0.45, n_neighbors_ignored=2, **kwargs):
        self.client = client
        self.filename = filename
        trajectory = md.load(filename, **kwargs)

        self.frames = range(len(trajectory))
        self.kwargs = kwargs

        ContactObject.__init__(self, trajectory.topology, query, haystack,
                               cutoff, n_neighbors_ignored)

        freq = dask_run(trajectory, client, self.run_info)
        self._n_frames = freq.n_frames
        self._atom_contacts = freq._atom_contacts
        self._residue_contacts = freq._residue_contacts

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


