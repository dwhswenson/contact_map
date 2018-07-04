from .utils import *

from contact_map.concurrence import *
from contact_map import ContactFrequency

def setup_module():
    global traj, contacts
    traj = md.load(find_testfile("concurrence.pdb"))
    query = traj.topology.select("resSeq 3")
    haystack = traj.topology.select("resSeq 1 to 2")
    # note that this includes *all* atoms
    contacts = ContactFrequency(traj, query, haystack, cutoff=0.051,
                                n_neighbors_ignored=0)

class TestAtomContactConcurrence(object):
    def setup(self):
        pass

    def test_default_labels(self):
        pass

    def test_get_items(self):
        pass

    def test_values(self):
        pass

class TestResidueContactConcurrence(object):
    def setup(self):
        self.heavy_contact_concurrence = ResidueContactConcurrence(
            trajectory=traj,
            residue_contacts=contacts.residue_contacts.most_common(),
            cutoff=0.051
        )

        self.all_contact_concurrence = ResidueContactConcurrence(
            trajectory=traj,
            residue_contacts=contacts.residue_contacts.most_common(),
            cutoff=0.051,
            select=""
        )

    @pytest.mark.parametrize('conc_type', ('heavy', 'all'))
    def test_default_labels(self, conc_type):
        concurrence = {'heavy': self.heavy_contact_concurrence,
                       'all': self.all_contact_concurrence}[conc_type]
        residue_labels = ['[AAA1, LLL3]', '[BBB2, LLL3]',
                          '[LLL3, AAA1]', '[LLL3, BBB2]']

        assert len(concurrence.labels) == 2
        for label in concurrence.labels:
            assert label in residue_labels

    def test_get_items(self):
        pass

    @pytest.mark.parametrize('conc_type', ('heavy', 'all'))
    def test_getitem(self, conc_type):
        concurrence = {'heavy': self.heavy_contact_concurrence,
                       'all': self.all_contact_concurrence}[conc_type]
        label_to_pair = {'[AAA1, LLL3]': 'AL',
                         '[LLL3, AAA1]': 'AL',
                         '[BBB2, LLL3]': 'BL',
                         '[LLL3, BBB2]': 'BL'}
        pair_to_expected = {
            'heavy': {
                'AL': [True, True, False, False, False],
                'BL': [True, False, False, False, False]
            },
            'all': {
                'AL': [True, True, True, True, False],
                'BL': [True, False, False, True, False]
            }
        }[conc_type]
        for label in concurrence.labels:
            values = concurrence[label]
            pair = label_to_pair[label]
            expected_values = pair_to_expected[pair]
            assert values == expected_values


class TestConcurrencePlotter(object):
    def setup(self):
        pass

    def test_x_values(self):
        pass

    def test_plot(self):
        # SMOKE TEST
        pass
