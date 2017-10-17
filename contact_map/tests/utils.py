import os
from pkg_resources import resource_filename

from numpy.testing import assert_array_equal
import pytest

def find_testfile(fname):
    return resource_filename('contact_map', os.path.join('tests', fname))

