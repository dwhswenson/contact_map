import os
from pkg_resources import resource_filename

def test_file(fname):
    return resource_filename('contact_map', os.path.join('tests', fname))

