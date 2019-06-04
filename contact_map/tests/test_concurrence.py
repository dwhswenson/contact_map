# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *

from contact_map.concurrence import *
from contact_map import ContactFrequency

# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import
def setup_module():
    global traj, contacts
    traj = md.load(find_testfile("concurrence.pdb"))
    query = traj.topology.select("resSeq 3")
    haystack = traj.topology.select("resSeq 1 to 2")
    # note that this includes *all* atoms
    contacts = ContactFrequency(traj, query, haystack, cutoff=0.051,
                                n_neighbors_ignored=0)


@pytest.mark.parametrize("contact_type", ('atoms', 'residues'))
def test_regularize_contact_input(contact_type):
    from contact_map.concurrence import _regularize_contact_input \
            as regularize
    most_common = {
        'atoms': contacts.atom_contacts.most_common(),
        'residues': contacts.residue_contacts.most_common()
    }[contact_type]
    contact_count = {
        'atoms': contacts.atom_contacts,
        'residues': contacts.residue_contacts
    }[contact_type]
    assert regularize(most_common, contact_type) == most_common
    assert regularize(contact_count, contact_type) == most_common
    assert regularize(contacts, contact_type) == most_common

def test_regularize_contact_input_bad_type():
    from contact_map.concurrence import _regularize_contact_input \
            as regularize
    with pytest.raises(RuntimeError):
        regularize(contacts, "foo")


class ContactConcurrenceTester(object):
    def _test_default_labels(self, concurrence):
        assert len(concurrence.labels) == len(self.labels) / 2
        for label in concurrence.labels:
            assert label in self.labels

    def _test_set_labels(self, concurrence, expected):
        new_labels = [self.label_to_pair[label]
                      for label in concurrence.labels]
        concurrence.set_labels(new_labels)
        for label in new_labels:
            assert concurrence[label] == expected[label]

    def _test_getitem(self, concurrence, pair_to_expected):
        for label in concurrence.labels:
            values = concurrence[label]
            pair = self.label_to_pair[label]
            expected_values = pair_to_expected[pair]
            assert values == expected_values


class TestAtomContactConcurrence(ContactConcurrenceTester):
    def setup(self):
        self.concurrence = AtomContactConcurrence(
            trajectory=traj,
            atom_contacts=contacts.atom_contacts.most_common(),
            cutoff=0.051
        )
        # dupes each direction until we have better way to handle frozensets
        self.label_to_pair = {'[AAA1-H, LLL3-H]': 'AH-LH',
                              '[LLL3-H, AAA1-H]': 'AH-LH',
                              '[AAA1-C1, LLL3-C1]': 'AC1-LC1',
                              '[LLL3-C1, AAA1-C1]': 'AC1-LC1',
                              '[BBB2-H, LLL3-C2]': 'BH-LC2',
                              '[LLL3-C2, BBB2-H]': 'BH-LC2',
                              '[AAA1-C2, LLL3-C2]': 'AC2-LC2',
                              '[LLL3-C2, AAA1-C2]': 'AC2-LC2',
                              '[AAA1-C2, LLL3-C1]': 'AC2-LC1',
                              '[LLL3-C1, AAA1-C2]': 'AC2-LC1',
                              '[BBB2-C2, LLL3-C2]': 'BC2-LC2',
                              '[LLL3-C2, BBB2-C2]': 'BC2-LC2'}
        self.labels = list(self.label_to_pair.keys())
        self.pair_to_expected = {
            'AH-LH': [False, False, True, True, False],
            'AC1-LC1': [False, True, False, False, False],
            'BH-LC2': [False, False, False, True, False],
            'AC2-LC2': [False, True, False, False, False],
            'AC2-LC1': [True, False, False, False, False],
            'BC2-LC2': [True, False, False, False, False]
        }

    def test_default_labels(self):
        self._test_default_labels(self.concurrence)

    def test_getitem(self):
        self._test_getitem(self.concurrence, self.pair_to_expected)

    def test_set_labels(self):
        self._test_set_labels(self.concurrence, self.pair_to_expected)


class TestResidueContactConcurrence(ContactConcurrenceTester):
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
        self.label_to_pair = {'[AAA1, LLL3]': 'AL',
                              '[LLL3, AAA1]': 'AL',
                              '[BBB2, LLL3]': 'BL',
                              '[LLL3, BBB2]': 'BL'}
        self.labels = list(self.label_to_pair.keys())
        self.pair_to_expected = {
            'heavy': {
                'AL': [True, True, False, False, False],
                'BL': [True, False, False, False, False]
            },
            'all': {
                'AL': [True, True, True, True, False],
                'BL': [True, False, False, True, False]
            }
        }

    def _get_concurrence(self, conc_type):
        return {'heavy': self.heavy_contact_concurrence,
                'all': self.all_contact_concurrence}[conc_type]


    @pytest.mark.parametrize('conc_type', ('heavy', 'all'))
    def test_default_labels(self, conc_type):
        concurrence = self._get_concurrence(conc_type)
        self._test_default_labels(concurrence)

    def test_set_labels(self):
        # only run this for the heavy atom concurrence
        concurrence = self.heavy_contact_concurrence
        expected = self.pair_to_expected['heavy']
        self._test_set_labels(concurrence, expected)

    @pytest.mark.parametrize('conc_type', ('heavy', 'all'))
    def test_getitem(self, conc_type):
        concurrence = self._get_concurrence(conc_type)
        pair_to_expected = self.pair_to_expected[conc_type]
        self._test_getitem(concurrence, pair_to_expected)


class TestConcurrencePlotter(object):
    def setup(self):
        self.concurrence = ResidueContactConcurrence(
            trajectory=traj,
            residue_contacts=contacts.residue_contacts.most_common(),
            cutoff=0.051
        )
        self.plotter = ConcurrencePlotter(self.concurrence)

    def test_x_values(self):
        time_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert self.plotter.x_values == [0, 1, 2, 3, 4]
        self.plotter.x_values = time_values
        assert self.plotter.x_values == time_values

    def test_get_concurrence_labels_given(self):
        alpha_labels = ['a', 'b']
        labels = self.plotter.get_concurrence_labels(self.concurrence,
                                                     labels=alpha_labels)
        assert labels == alpha_labels

    def test_get_concurrence_labels_default(self):
        labels = self.plotter.get_concurrence_labels(self.concurrence)
        assert labels == self.concurrence.labels

    def test_get_concurrence_label_none_in_concurrence(self):
        numeric_labels = ['0', '1']
        self.concurrence.labels = None
        labels = self.plotter.get_concurrence_labels(self.concurrence)
        assert labels == numeric_labels

    def test_plot(self):
        # SMOKE TEST ONLY
        pytest.importorskip('matplotlib.pyplot')
        self.plotter.plot()
