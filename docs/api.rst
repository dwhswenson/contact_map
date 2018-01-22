.. _api:


#############
API Reference
#############

Contact maps
------------

.. currentmodule:: contact_map

.. autosummary:: 
    :toctree: api/generated/

    ContactCount
    ContactMap
    ContactFrequency
    ContactDifference

Minimum Distance (and related)
------------------------------

.. .. currentmodule:: min_dist

.. autosummary::
    :toctree: api/generated/

    MinimumDistanceCounter
    NearestAtoms


Parallelization of ``ContactFrequency``
---------------------------------------

.. autosummary::
    :toctree: api/generated/

    frequency_task
    DaskContactFrequency

-----


API naming conventions
----------------------

There are several terms that are used throughout the API which are not
completely standard. Understanding them, and how we use them, will make it
much easier to understand the code.

.. note::

    This section does not discuss the code style conventions we use, only
    the choice of specific words to mean specific things outside the normal
    scientific usage. For the code style, see the (to-be-written) developer
    documentation (or just use PEP8).

Query/Haystack
~~~~~~~~~~~~~~

Many functions in the API take the lists ``query`` and ``haystack`` as
input. This nomenclature follows usage in MDTraj. These are lists of atom
indices used in the contact search. Every pair will include one atom from
``query`` and one atom from ``haystack``. In principle, the two lists are
interchangeable. However, there are cases where the implementation will be
faster if the ``query`` is the smaller of the two lists.

Index/idx
~~~~~~~~~

Most of our return values are in terms of MDTraj ``Atom`` and ``Residue``
objects. This is because these are more readable, and provide the user with
immediate access to useful context. However, there are times that what we
really want is the atom or residue index number. For this, we include the
``idx`` suffix (e.g., ``most_common_atoms_idx``). Note that these indices
start from 0; this can be confusing when comparing to PDB entries where
indexing is from 1.

Most common
~~~~~~~~~~~

Several methods begin with ``most_common``. The behavior for this is
inspired by the behavior of :meth:`collections.Counter.most_common`, which
returns elements and there counts ordered from most to least. Note that,
unlike the original, we usually do not implement a way to only return the
first ``n`` results (although this may be added later).
