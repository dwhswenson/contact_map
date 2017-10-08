[![Linux Build Status](https://travis-ci.org/dwhswenson/contact_map.svg?branch=master)](https://travis-ci.org/dwhswenson/contact_map)
[![Windows Build status](https://ci.appveyor.com/api/projects/status/em3fo96sjrg2vmcc/branch/master?svg=true)](https://ci.appveyor.com/project/dwhswenson/contact-map/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/dwhswenson/contact_map/badge.svg?branch=master)](https://coveralls.io/github/dwhswenson/contact_map?branch=master)
[![codebeat badge](https://codebeat.co/badges/c7fb604a-35a8-4ccf-afea-18d6bd494726)](https://codebeat.co/projects/github-com-dwhswenson-contact_map-master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/f7f3cf53698e4655ac8895f13fa5dea6)](https://www.codacy.com/app/dwhswenson/contact_map?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dwhswenson/contact_map&amp;utm_campaign=Badge_Grade)
[![Code Climate](https://codeclimate.com/github/dwhswenson/contact_map/badges/gpa.svg)](https://codeclimate.com/github/dwhswenson/contact_map)
[![PyPI](https://img.shields.io/pypi/v/contact-map.svg)](https://pypi.python.org/pypi/contact-map/)

# Contact Maps

This package provides tools for analyzing and exploring contacts
(residue-residue and atom-atom) in a molecular dynamics simulation. It
builds on the excellent tools provided by [MDTraj](http://mdtraj.org).

## Installation

TODO: Neither of these installs actually work until we have a release! Use
developer build until then.

The easiest way to install is with `conda`. Conda is a powerful package and
environment management system; if you do not already have a highly
customized Python environment, we recommend starting by installing `conda`,
either in the [full anaconda
distribution](https://www.anaconda.com/download/) or the [smaller-footprint
miniconda](https://conda.io/miniconda.html).

```bash
conda install -c conda-forge contact_map
```

The second-easiest install, which will work on any Python setup (conda or
not) is to install via `pip`. Because of some weirdness in how `pip` handles
packages (like MDTraj) that depend on Numpy, you should install Cython and
Numpy separately, so the whole install is:

```bash
pip install cython
pip install numpy
pip install contact_map
```

If you already have Numpy installed, you may need to re-install it with `pip
install -U --force-reinstall numpy`.


### Installing a developer build

If you plan to work with the source, or if you want to stay on the bleeding
edge, you can install a version so that your downloaded/cloned version of
this git repository 

#### 1. Download or clone the respository

#### 2. Install the requirements

There are two ways to install the requirements. You can either use `conda`
or `pip`. If you don't already have a significant Python installation, we
recommend using `conda` for all your Python needs. However, if you *do*
already have a significant Python installation, you may find it annoying to
migrate everything over to `conda`. In that case, you might use the
`pip`-based installation. Note that, due to some challenges in installing
MDTraj, only the `conda`-based install has been tested on Windows.

Using `conda`:

```bash
conda install -y --file requirements.txt
```

Or install using `pip`:

```bash
pip install -r requirements.txt
```

Note that MDTraj can be a bit particular about the numpy that it uses. If,
in the first step of the `pip` install, MDTraj complains, try running the
following commands:

```bash
pip install cython
pip install --upgrade --force-reinstall numpy
```

#### 3. Install the package

##
