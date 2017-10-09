.. _installation:

############
Installation
############

If you're just planning to use the code, you'll want to perform a basic
installation. If you're planning to develop for the code, or if you want to
stay on the bleeding edge, then you should perform a developer installation.

Basic Installation
==================

There are two recommended approaches for a basic installation:
``conda``-based, or ``pip``-based. Using ``conda`` is much easier, and will
continue to be easier for anything else you install. However, the
disadvantage is that you must put your entire Python environment under
``conda``. If you already have a highly customized Python environment, you
might prefer the ``pip`` install. But otherwise, we highly recommend
installing ``conda``, either using the `full Ananconda distribution
<https://www.anaconda.com/download/>`_ or the `smaller-footprint miniconda
<https://conda.io/miniconda.html>`_. Once ``conda`` is installed and in
your path, installation is as simple as:

.. code:: bash

    conda install -c conda-forge contact_map

which tells ``conda`` to get ``contact_map`` from the `conda-forge
<https://conda-forge.org/>`_ channel, which manages our ``conda``-based
installation recipe.

If you would prefer to use ``pip``, that takes a few extra steps, but will
work on any Python setup (``conda`` or not). Because of some weirdness in
how ``pip`` handles packages (such as MDTraj) that have a particular types
of requirements from Numpy, you should install Cython
and Numpy separately, so the whole install is:

.. code:: bash

    pip install cython
    pip install numpy
    pip install contact_map

If you already have Numpy installed, you may need to re-install it with
``pip install -U --force-reinstall numpy``. Note that some systems may
require you to preface ``pip install`` commands with ``sudo`` (depending on
where Python keeps its packages).

Developer installation
======================

If you plan to work with the source, or if you want to stay on the bleeding
edge, you can install a version so that your downloaded/cloned version of
this git repository is the live code your Python interpreter sees. We call
that a "developer installation."
