import itertools
import mdtraj as md
import numpy as np

class Concurrence(object):
    def __init__(self, values, labels=None):
        self.values = values
        self.labels = labels

    @property
    def lifetimes(self):
        pass

    def set_labels(self, labels):
        self.labels = labels

    def __getitem__(self, label):
        idx = self.labels.index(label)
        return self.values[idx]

    def coincidence(self, label_list):
        this_list = np.asarray(self[label_list[0]])
        coincidence_list = this_list
        norm_sq = sum(this_list)
        for label in label_list[1:]:
            this_list = np.asarray(self[label])
            coincidence_list &= this_list
            norm_sq *= sum(this_list)

        return sum(coincidence_list) / np.sqrt(norm_sq)



class AtomContactConcurrence(Concurrence):
    def __init__(self, trajectory, atom_contacts, cutoff=0.45):
        atom_pairs = [[contact[0][0].index, contact[0][1].index]
                       for contact in atom_contacts]
        labels = [str(contact[0]) for contact in atom_contacts]
        distances = md.compute_distances(trajectory, atom_pairs=atom_pairs)
        vector_f = np.vectorize(lambda d: d < cutoff)
        values = zip(*vector_f(distances))
        super(AtomContactConcurrence, self).__init__(values=values,
                                                     labels=labels)

class ResidueContactConcurrence(Concurrence):
    def __init__(self, trajectory, residue_contacts, cutoff=0.45,
                 select="and not symbol == 'H'"):
        residue_pairs = [[contact[0][0], contact[0][1]]
                         for contact in residue_contacts]
        labels = [str(contact[0]) for contact in residue_contacts]
        values = []
        for res_A, res_B in residue_pairs:
            atoms_A = trajectory.topology.select("resid " + str(res_A.index)
                                                 + " " + select)
            atoms_B = trajectory.topology.select("resid " + str(res_B.index)
                                                 + " " + select)
            atom_pairs = itertools.product(atoms_A, atoms_B)
            distances = md.compute_distances(trajectory,
                                             atom_pairs=atom_pairs)
            min_dists = [min(dists) for dists in distances]
            values.append(map(lambda d: d < cutoff, min_dists))

        super(ResidueContactConcurrence, self).__init__(values=values,
                                                        labels=labels)

def plot_concurrence(concurrence, labels=None, x_values=None):
    import matplotlib.pyplot as plt
    if x_values is None:
        x_values = range(len(concurrence.values[0]))
    if labels is None:
        if concurrence.labels is not None:
            labels = concurrence.labels
        else:
            labels = [str(i) for i in range(len(values))]

    y_val = -1.0
    for label, val_set in zip(labels, concurrence.values):
        x_vals = [x for (x, y) in zip(x_values, val_set) if y]
        plt.plot(x_vals, [y_val] * len(x_vals), '.', markersize=1, label=label)
        y_val -= 1.0

    plt.ylim(ymax=0.0)
    plt.xlim(xmin=min(x_values), xmax=max(x_values))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
