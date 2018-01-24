"""
Task-based implementation of :class:`.ContactFrequency`.

The overall algorithm is:

1. Identify how we're going to slice up the trajectory into task-based
   chunks (:meth:`block_slices`, :meth:`default_slices`)
2. On each node
    a. Load the trajectory segment (:meth:`load_trajectory_task`)
    b. Run the analysis on the segment (:meth:`map_task`)
3. Once all the results have been collected, combine them
   (:meth:`reduce_all_results`)

Notes
-----
Includes versions where messages are Python objects and versions (labelled
with _json) where messages have been JSON-serialized. However, we don't yet
have a solution for JSON serialization of MDTraj objects, so if JSON
serialization is the communication method, the loading of the trajectory and
the calculation of the contacts must be combined into a single task.
"""

import mdtraj as md
from contact_map import ContactFrequency

def block_slices(n_total, n_per_block):
    """Determine slices for splitting the input array.

    Parameters
    ----------
    n_total : int
        total length of array
    n_per_block : int
        maximum number of items per block

    Returns
    -------
    list of slice
        slices to be applied to the array
    """
    n_full_blocks = n_total // n_per_block
    slices = [slice(i*n_per_block, (i+1)*n_per_block)
              for i in range(n_full_blocks)]
    if n_total % n_per_block:
        slices.append(slice(n_full_blocks*n_per_block, n_total))
    return slices

def default_slices(n_total, n_workers):
    """Calculate default slices from number of workers.

    Default behavior is (approximately) one task per worker.

    Parameters
    ----------
    n_total : int
        total number of items in array
    n_workers : int
        number of workers

    Returns
    -------
    list of slice
        slices to be applied to the array
    """
    n_frames_per_task = max(1, n_total // n_workers)
    return block_slices(n_total, n_frames_per_task)


def load_trajectory_task(subslice, file_name, **kwargs):
    """
    Task for loading file. Reordered for to take per-task variable first.

    Parameters
    ----------
    subslice : slice
        the slice of the trajectory to use
    file_name : str
        trajectory file name
    kwargs :
        other parameters to mdtraj.load

    Returns
    -------
    md.Trajectory :
        subtrajectory for this slice
    """
    return md.load(file_name, **kwargs)[subslice]

def map_task(subtrajectory, parameters):
    """Task to be mapped to all subtrajectories. Run ContactFrequency

    Parameters
    ----------
    subtrajectory : mdtraj.Trajectory
        single trajectory segment to calculate ContactFrequency for
    parameters : dict
        kwargs-style dict for the :class:`.ContactFrequency` object

    Returns
    -------
    :class:`.ContactFrequency` :
        contact frequency for the subtrajectory
    """
    return ContactFrequency(subtrajectory, **parameters)

def reduce_all_results(contacts):
    """Combine multiple :class:`.ContactFrequency` objects into one

    Parameters
    ----------
    contacts : iterable of :class:`.ContactFrequency`
        the individual (partial) contact frequencies

    Returns
    -------
    :class:`.ContactFrequency` :
        total of all input contact frequencies (summing them)
    """
    accumulator = contacts[0]
    for contact in contacts[1:]:
        accumulator.add_contact_frequency(contact)
    return accumulator


def map_task_json(subtrajectory, parameters):
    """JSON-serialized version of :meth:`map_task`"""
    return map_task(subtrajectory, parameters).to_json()

def reduce_all_results_json(results_of_map):
    """JSON-serialized version of :meth:`reduce_all_results`"""
    contacts = [ContactFrequency.from_json(res) for res in results_of_map]
    return reduce_all_results(contacts)
