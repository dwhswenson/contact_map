# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *
from contact_map.dask_runner import *
from contact_map import ContactFrequency, ContactTrajectory
from collections.abc import Iterable
import mdtraj


def dask_setup_test_cluster(distributed, n_workers=4, n_attempts=3):
    """Set up a test cluster using dask.distributed. Try up to n_attempts
    times, and skip the test if all attempts fail.
    """
    cluster = None
    for _ in range(n_attempts):
        try:
            cluster = distributed.LocalCluster(n_workers=n_workers)
        except AttributeError:
            # should never get here, because should have already skipped
            pytest.skip("dask.distributed not installed")
        except distributed.TimeoutError:
            continue
        else:
            return cluster
    # only get here if all retries fail
    pytest.skip("Failed to set up distributed LocalCluster")


class TestDaskRunners(object):
    def setup(self):
        dask = pytest.importorskip('dask')  # pylint: disable=W0612
        distributed = pytest.importorskip('dask.distributed')
        self.distributed = distributed
        # Explicitly set only 4 workers on Travis instead of 31
        # Fix copied from https://github.com/spencerahill/aospy/pull/220/files
        self.cluster = dask_setup_test_cluster(distributed, n_workers=4)
        self.client = distributed.Client(self.cluster)
        self.filename = find_testfile("trajectory.pdb")

    def teardown(self):
        self.client.shutdown()

    @pytest.mark.parametrize("dask_cls", [DaskContactFrequency,
                                          DaskContactTrajectory])
    def test_dask_integration(self, dask_cls):
        dask_freq = dask_cls(self.client, self.filename, cutoff=0.075,
                             n_neighbors_ignored=0)
        if isinstance(dask_freq, ContactFrequency):
            assert dask_freq.n_frames == 5
        elif isinstance(dask_freq, ContactTrajectory):
            assert len(dask_freq) == 5

    def test_dask_atom_slice(self):
        dask_freq0 = DaskContactFrequency(self.client, self.filename,
                                          query=[3, 4],
                                          haystack=[6, 7], cutoff=0.075,
                                          n_neighbors_ignored=0)
        self.client.close()
        assert dask_freq0.n_frames == 5
        self.client = self.distributed.Client(self.cluster)
        # Set the slicing of contact frequency (used in the frqeuency task)
        # to False
        ContactFrequency._class_use_atom_slice = False
        dask_freq1 = DaskContactFrequency(self.client,
                                          self.filename, query=[3, 4],
                                          haystack=[6, 7], cutoff=0.075,
                                          n_neighbors_ignored=0)
        assert dask_freq0._use_atom_slice is True
        assert dask_freq1._use_atom_slice is False
        assert dask_freq0 == dask_freq1

    @pytest.mark.parametrize("dask_cls, norm_cls",[
        (DaskContactFrequency, ContactFrequency),
        (DaskContactTrajectory, ContactTrajectory)])
    def test_answer_equal(self, dask_cls, norm_cls):
        trj = mdtraj.load(self.filename)
        norm_result = norm_cls(trj)
        dask_result = dask_cls(self.client, self.filename)
        if isinstance(dask_result, Iterable):
            for i, j in zip(dask_result, norm_result):
                assert i.atom_contacts._counter == j.atom_contacts._counter
                assert (i.residue_contacts._counter ==
                        j.residue_contacts._counter)
        else:
            assert (dask_result.atom_contacts._counter ==
                    norm_result.atom_contacts._counter)
            assert (dask_result.residue_contacts._counter ==
                    norm_result.residue_contacts._counter)
