import mdtraj as md
from contact_map import NearestAtoms
import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--cutoff', default=1.0,
                        help="cutoff distance (nm)")
    parser.add_argument('-N', default=10,
                        help="""Report this number of atom pairs and
                        distances; 0 indicates all atoms. Default 10""")

if __name__ == "__main__":
    parser = parser()
    opts = parser.parse_args()
    results = NearestAtoms(opts.filename, opts.cutoff).sorted_distances
    n_max = opts.N if opts.N > 0 else len(results)
    for (atom_i, atom_j, dist) in results[:n_max]:
        print atom_i, atom_j, dist

