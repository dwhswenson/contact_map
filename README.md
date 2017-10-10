[![Linux Build Status](https://travis-ci.org/dwhswenson/contact_map.svg?branch=master)](https://travis-ci.org/dwhswenson/contact_map)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/em3fo96sjrg2vmcc/branch/master?svg=true)](https://ci.appveyor.com/project/dwhswenson/contact-map/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/dwhswenson/contact_map/badge.svg?branch=master)](https://coveralls.io/github/dwhswenson/contact_map?branch=master)
[![Documentation Status](https://readthedocs.org/projects/contact-map/badge/?version=latest)](http://contact-map.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/contact-map.svg)](https://pypi.python.org/pypi/contact-map/)
[![conda-forge](https://img.shields.io/conda/v/conda-forge/contact_map.svg)](https://github.com/conda-forge/contact_map-feedstock)

[![codebeat badge](https://codebeat.co/badges/c7fb604a-35a8-4ccf-afea-18d6bd494726)](https://codebeat.co/projects/github-com-dwhswenson-contact_map-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f7f3cf53698e4655ac8895f13fa5dea6)](https://www.codacy.com/app/dwhswenson/contact_map?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dwhswenson/contact_map&amp;utm_campaign=Badge_Grade)
[![Code Climate](https://codeclimate.com/github/dwhswenson/contact_map/badges/gpa.svg)](https://codeclimate.com/github/dwhswenson/contact_map)

# Contact Maps

This package provides tools for analyzing and exploring contacts
(residue-residue and atom-atom) in a molecular dynamics simulation. It
builds on the excellent tools provided by [MDTraj](http://mdtraj.org).

## Installation

The easiest way to install is with `conda`. Conda is a powerful package and
environment management system; if you do not already have a highly
customized Python environment, we recommend starting by installing `conda`,
either in the [full anaconda
distribution](https://www.anaconda.com/download/) or the [smaller-footprint
miniconda](https://conda.io/miniconda.html). This package is distributed
through the [conda-forge](http://conda-forge.org) channel; install it with:

```bash
conda install -c conda-forge contact_map
```

If you don't want to use `conda`, you can also use `pip` (via a more
complicated process) or do a developer install. See the [installation
documentation](http://contact-map.readthedocs.io/en/latest/installing.html)
for details.
