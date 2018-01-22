import os
import numpy as np
from pkg_resources import resource_filename

from numpy.testing import assert_array_equal, assert_allclose
import pytest

def find_testfile(fname):
    return resource_filename('contact_map', os.path.join('tests', fname))

def zero_to_nan(input_arr):
    arr = input_arr.copy()
    arr[arr == 0.0] = np.nan
    return arr
