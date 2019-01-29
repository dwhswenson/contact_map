import itertools
import mdtraj as md
import numpy as np

import contact_map
from .contact_map import ContactObject

try:
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MATPLOTLIB = False
else:
    HAS_MATPLOTLIB = True


class Concurrence(object):
    """Superclass for contact concurrence objects.

    Contact concurrences measure what contacts occur simultaneously in a
    trajectory. When defining states, one usually wants to characterize
    based on multiple contacts that are made simultaneously; contact
    concurrences makes it easier to identify those.

    Parameters
    ----------
        values : list of list of bool
            the whether a contact is present for each contact pair at each
            point in time; inner list is length number of frames, outer list
            in length number of (included) contacts
        labels : list of string
            labels for each contact pair
    """
    def __init__(self, values, labels=None):
        self.values = values
        self.labels = labels

    # @property
    # def lifetimes(self):
        # pass

    def set_labels(self, labels):
        """Set the contact labels

        Parameters
        ----------
        labels : list of string
            labels for each contact pair
        """
        self.labels = labels

    def __getitem__(self, label):
        idx = self.labels.index(label)
        return self.values[idx]

    # temporarily removed until we find a good metric here; this metric did
    # not seem optimal and I stopped using it, so remove from code before
    # release (can add back in later)
    # def coincidence(self, label_list):
        # this_list = np.asarray(self[label_list[0]])
        # coincidence_list = this_list
        # norm_sq = sum(this_list)
        # for label in label_list[1:]:
            # this_list = np.asarray(self[label])
            # coincidence_list &= this_list
            # norm_sq *= sum(this_list)

        # return sum(coincidence_list) / np.sqrt(norm_sq)

def _regularize_contact_input(contact_input, atom_or_res):
    """Clean input for concurrence objects.

    The allowed inputs are the :class:`.ContactFrequency`, or the
    :class:`.ContactObject` coming from the ``.atom_contacts`` or
    ``.residue_contacts`` attribute of the contact frequency, or the list
    coming from the ``.most_common()`` method for the
    :class:`.ContactObject`.

    Parameters
    ----------
    contact_input : many possible types; see method description
        input to the contact concurrences
    atom_or_res : string
        whether to treat this as an atom-based or residue-based contact;
        allowed values are "atom", "res", and "residue"

    Returns
    -------
    list :
        list in the format of ``ContactCount.most_common()``
    """
    if isinstance(contact_input, ContactObject):
        contact_input = contact_input.contacts[atom_or_res]

    if isinstance(contact_input, contact_map.ContactCount):
        contact_input = contact_input.most_common()

    return contact_input


class AtomContactConcurrence(Concurrence):
    """Contact concurrences for atom contacts.

    Parameters
    ----------
    trajectory : :class:`mdtraj.Trajectory`
        the trajectory to analyze
    atom_contacts : list
        output from ``contact_map.atom_contacts.most_common()``
    cutoff : float
        cutoff, in nm. Should be the same as used in the contact map.
    """
    def __init__(self, trajectory, atom_contacts, cutoff=0.45):
        atom_contacts = _regularize_contact_input(atom_contacts, "atom")
        atom_pairs = [[contact[0][0].index, contact[0][1].index]
                      for contact in atom_contacts]
        labels = [str(contact[0]) for contact in atom_contacts]
        distances = md.compute_distances(trajectory, atom_pairs=atom_pairs)
        vector_f = np.vectorize(lambda d: d < cutoff)
        # transpose because distances is ndarray shape (n_frames,
        # n_contacts); values should be list shape (n_contacts, n_frames)
        values = vector_f(distances).T.tolist()
        super(AtomContactConcurrence, self).__init__(values=values,
                                                     labels=labels)


class ResidueContactConcurrence(Concurrence):
    """Contact concurrences for residue contacts.

    Parameters
    ----------
    trajectory : :class:`mdtraj.Trajectory`
        the trajectory to analyze
    residue_contacts : list
        output from ``contact_map.residue_contacts.most_common()``
    cutoff : float
        cutoff, in nm. Should be the same as used in the contact map.
    select : string
        additional atom selection string for MDTraj; defaults to "and symbol
        != 'H'"
    """
    def __init__(self, trajectory, residue_contacts, cutoff=0.45,
                 select="and symbol != 'H'"):
        residue_contacts = _regularize_contact_input(residue_contacts,
                                                     "residue")
        residue_pairs = [[contact[0][0], contact[0][1]]
                         for contact in residue_contacts]
        labels = [str(contact[0]) for contact in residue_contacts]
        values = []
        select_residue = lambda idx: trajectory.topology.select(
            "resid " + str(idx) + " " + select
        )
        for res_A, res_B in residue_pairs:
            atoms_A = select_residue(res_A.index)
            atoms_B = select_residue(res_B.index)
            atom_pairs = itertools.product(atoms_A, atoms_B)
            distances = md.compute_distances(trajectory,
                                             atom_pairs=atom_pairs)
            min_dists = [min(dists) for dists in distances]
            values.append([d < cutoff for d in min_dists])

        super(ResidueContactConcurrence, self).__init__(values=values,
                                                        labels=labels)


class ConcurrencePlotter(object):
    """Plot manager for contact concurrences.

    Parameters
    ----------
    concurrence : :class:`.Concurrence`
        concurrence to plot; default None allows to override later
    labels : list of string
        labels for the contact pairs, default None will use concurrence
        labels if available, integers if not
    x_values : list of numeric
        values to use for the time axis; default None uses integers starting
        at 0 (can be used to assign the actual simulation time to the
        x-axis)
    """
    def __init__(self, concurrence=None, labels=None, x_values=None):
        self.concurrence = concurrence
        self.labels = self.get_concurrence_labels(concurrence, labels)
        self._x_values = x_values

    @staticmethod
    def get_concurrence_labels(concurrence, labels=None):
        """Extract labels for contact from a concurrence object

        If ``labels`` is given, that is returned. Otherwise, the
        ``concurrence`` is checked for labels, and those are used. If those
        are also not available, string forms of integers starting with 0 are
        returned.


        Parameters
        ----------
        concurrence : :class:`.Concurrence`
            concurrence, which may have label information
        labels : list of string
            labels to use for contacts (optional)

        Returns
        -------
        list of string
            labels to use for contacts
        """
        if labels is None:
            if concurrence and concurrence.labels is not None:
                labels = concurrence.labels
            else:
                labels = [str(i) for i in range(len(concurrence.values))]
        return labels

    @property
    def x_values(self):
        """list : values to use for the x-axis (time)"""
        x_values = self._x_values
        if x_values is None:
            x_values = list(range(len(self.concurrence.values[0])))
        return x_values

    @x_values.setter
    def x_values(self, x_values):
        self._x_values = x_values

    def plot(self, concurrence=None, **kwargs):
        """Contact concurrence plot based on matplotlib

        Additional kwargs given here will be passed to the matplotlib
        ``Axes.plot()`` method.

        Parameters
        ----------
        concurrence : :class:`.Concurrence`
            optional; default None uses ``self.concurrence``; this allows
            one to override the use of ``self.concurrence``

        Returns
        -------
        fig : :class:`.matplotlib.Figure`
        ax : :class:`.matplotlib.Axes`
        lgd: :class:`.matplotlib.legend.Legend`
            objects for matplotlib-based plot of contact concurrences
        """
        if not HAS_MATPLOTLIB:  # pragma: no cover
            raise ImportError("matplotlib not installed")
        if concurrence is None:
            concurrence = self.concurrence
        labels = self.get_concurrence_labels(concurrence=concurrence)
        x_values = self.x_values

        fig = plt.figure(1)
        ax = fig.add_subplot(111)

        plot_kwargs = {'markersize': 1}
        plot_kwargs.update(kwargs)

        y_val = -1.0
        for label, val_set in zip(labels, concurrence.values):
            x_vals = [x for (x, y) in zip(x_values, val_set) if y]
            ax.plot(x_vals, [y_val] * len(x_vals), '.', label=label,
                    **plot_kwargs)
            y_val -= 1.0

        ax.set_ylim(top=0.0)
        ax.set_xlim(left=min(x_values), right=max(x_values))
        ax.set_yticks([])
        lgd = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        return (fig, ax, lgd)


def plot_concurrence(concurrence, labels=None, x_values=None, **kwargs):  # -no-cov-
    """
    Convenience function for concurrence plots.

    Additional kwargs given here will be passed to the matplotlib
    ``Axes.plot()`` method.

    Parameters
    ----------
    concurrence : :class:`.Concurrence`
        concurrence to be plotted
    labels: list of string
        labels for contacts (optional)
    x_values : list of float or list of int
        values to use for the x-axis

    See also
    --------
    :class:`.ConcurrencePlotter`
    """
    return ConcurrencePlotter(concurrence, labels, x_values).plot(**kwargs)
