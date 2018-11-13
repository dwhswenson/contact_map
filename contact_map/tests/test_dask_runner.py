
# pylint: disable=wildcard-import, missing-docstring, protected-access
# pylint: disable=attribute-defined-outside-init, invalid-name, no-self-use
# pylint: disable=wrong-import-order, unused-wildcard-import

from .utils import *
from contact_map.dask_runner import *

def dask_setup_test_cluster(distributed, n_workers=4, n_attempts=3):
    """Set up a test cluster using dask.distributed. Try up to n_attempts
    times, and skip the test if all attempts fail.
    """
    cluster = None
    for _ in range(n_attempts):
        try:
            cluster = distributed.LocalCluster(n_workers=n_workers)
        except distributed.TimeoutError:
            continue
        else:
            return cluster
    # only get here if all retries fail
    pytest.skip("Failed to set up distributed LocalCluster")


class TestDaskContactFrequency(object):
    def test_dask_integration(self):
        # this is an integration test to check that dask works
        dask = pytest.importorskip('dask')  # pylint: disable=W0612
        distributed = pytest.importorskip('dask.distributed')
        # Explicitly set only 4 workers on Travis instead of 31
        # Fix copied from https://github.com/spencerahill/aospy/pull/220/files
        cluster = dask_setup_test_cluster(distributed)
        client = distributed.Client(cluster)
        filename = find_testfile("trajectory.pdb")

        dask_freq = DaskContactFrequency(client, filename, cutoff=0.075,
                                         n_neighbors_ignored=0)
        client.close()
        assert dask_freq.n_frames == 5
