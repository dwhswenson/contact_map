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

This is a three-step process:

1. **Download or clone the repository.** If you plan to contribute changes
   back to the repository, please fork it on GitHub and then clone your
   fork. Otherwise, you can download or clone the main repository. You can
   follow `GitHub's instructions on how to do this
   <https://help.github.com/articles/fork-a-repo/>`_, and apply those steps
   to forking our repository at http://github.com/dwhswenson/contact_map.

2. **Install the requirements.** This can be done using either ``pip`` or
   ``conda``. First, change into the directory for the repository. You
   should see ``setup.py`` and ``requirements.txt`` (among many other
   things) in that directory. Using conda:
   
   .. code:: bash

      conda install -y --file requirements.txt

   Or, using ``pip``:

   .. code:: bash
       
      pip install cython
      pip install numpy
      pip install -r requirements.txt

   In some cases, you may need to add ``-U --force-reinstall`` to the Numpy
   step.

3. **Install the package.** Whether you get the requirements with ``pip`` or
   with ``conda``, you can install the package (again, from the directory
   containing ``setup.py``) with:

   .. code:: bash

      pip install -e .

   The ``-e`` means that the installation is "editable" (developer version;
   the stuff in this directory will be the live code your Python
   interpreted uses) and the ``.`` tells it to find ``setup.py`` in the
   current directory.


Testing your installation
=========================

However you have installed it, you should test that your installation works.
To do so, first check that the new package can be imported. This can be done
with

.. code:: bash

   python -c "import contact_map"

If your Python interpreter can find the newly-installed package, that should
exit without complaint.

For a more thorough check that everything works, you should run our test
suite. This can be done by installing ``pytest`` (using either ``pip`` or
``conda``) and then running the command:

.. code:: bash

   py.test --pyargs contact_map -v

This will run the tests on the installed version of ``contact_map``. All
tests should either pass or skip.
