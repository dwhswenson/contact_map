#/usr/bin/env python
from __future__ import print_function
import sys
import setup
import contact_map
from autorelease import DefaultCheckRunner, conda_recipe_version

repo_path = '.'
versions = {
    'package': contact_map.version.version,
    'setup.py': setup.PACKAGE_VERSION,
    'conda-recipe': conda_recipe_version('ci/conda-recipe/meta.yaml'),
}

RELEASE_BRANCHES = ['stable']

if __name__ == "__main__":
    checker = DefaultCheckRunner(
        versions=versions,
        setup=setup,
        repo_path='.'
    )
    checker.release_branches = RELEASE_BRANCHES

    tests = checker.select_tests_from_sysargs()
    n_fails = checker.run_as_test(tests)
