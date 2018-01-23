# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *
from .test_contact_map import traj

from contact_map.frequency_task import *
from contact_map import ContactFrequency

class TestSlicing(object):
    # tests for block_slices and default_slices
    @pytest.mark.parametrize("inputs, results", [
        ((100, 25),
         [slice(0, 25), slice(25, 50), slice(50, 75), slice(75, 100)]),
        ((85, 25),
         [slice(0, 25), slice(25, 50), slice(50, 75), slice(75, 85)])
    ])
    def test_block_slices(self, inputs, results):
        n_total, n_per_block = inputs
        assert block_slices(n_total, n_per_block) == results

    @pytest.mark.parametrize("inputs, results", [
        ((100, 4),
         [slice(0, 25), slice(25, 50), slice(50, 75), slice(75, 100)]),
        ((77, 3),
         [slice(0, 25), slice(25, 50), slice(50, 75), slice(75, 77)]),
        ((2, 20),
         [slice(0, 1), slice(1, 2)])
    ])
    def test_default_slice_even_split(self, inputs, results):
        n_total, n_workers = inputs
        assert default_slices(n_total, n_workers) == results

class TestTasks(object):
    def setup(self):
        self.contact_freq_0_4 = ContactFrequency(traj, cutoff=0.075,
                                                 n_neighbors_ignored=0,
                                                 frames=range(4))
        self.contact_freq_4 = ContactFrequency(traj, cutoff=0.075,
                                               n_neighbors_ignored=0,
                                               frames=[4])
        self.total_contact_freq = ContactFrequency(traj, cutoff=0.075,
                                                   n_neighbors_ignored=0)
        self.parameters = {'cutoff': 0.075, 'n_neighbors_ignored': 0}

    def test_load_trajectory_task(self):
        subslice = slice(0, 4)
        file_name = find_testfile("trajectory.pdb")
        trajectory = load_trajectory_task(subslice, file_name)
        assert trajectory.xyz.shape == (4, 10, 3)

    def test_map_task(self):
        trajectory = traj[:4]
        mapped = map_task(trajectory, parameters=self.parameters)
        assert mapped == self.contact_freq_0_4

    def test_reduce_task(self):
        reduced = reduce_all_results([self.contact_freq_0_4,
                                      self.contact_freq_4])
        assert reduced == self.total_contact_freq

    def test_map_task_json(self):
        # check the json objects by converting them back to full objects
        trajectory = traj[:4]
        mapped = map_task_json(trajectory, parameters=self.parameters)
        assert ContactFrequency.from_json(mapped) == self.contact_freq_0_4

    def test_reduce_all_results_json(self):
        reduced = reduce_all_results_json([self.contact_freq_0_4.to_json(),
                                           self.contact_freq_4.to_json()])
        assert reduced == self.total_contact_freq

    def test_integration_object_based(self):
        file_name = find_testfile("trajectory.pdb")
        slices = default_slices(len(traj), n_workers=3)
        trajs = [load_trajectory_task(subslice=sl,
                                      file_name=file_name)
                 for sl in slices]
        mapped = [map_task(subtraj, self.parameters) for subtraj in trajs]
        result = reduce_all_results(mapped)
        assert result == self.total_contact_freq

    def test_integration_json_based(self):
        file_name = find_testfile("trajectory.pdb")
        slices = default_slices(len(traj), n_workers=3)
        trajs = [load_trajectory_task(subslice=sl,
                                      file_name=file_name)
                 for sl in slices]
        mapped = [map_task_json(subtraj, self.parameters)
                  for subtraj in trajs]
        result = reduce_all_results_json(mapped)
        assert result == self.total_contact_freq

