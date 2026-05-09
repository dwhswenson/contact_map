import os
from importlib.resources import files

# pylint: disable=unused-import

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import pytest

def find_testfile(fname):
    return os.fspath(files(__package__).joinpath(fname))

def zero_to_nan(input_arr):
    arr = input_arr.copy()
    arr[arr == 0.0] = np.nan
    return arr
