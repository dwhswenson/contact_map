
{% if "dev" in release %}                                            
{% set version = "master" %}                                                    
{% else %}                                                                      
{% set version = "v"+version %}                                      
{% endif %}  

User Guide/Examples
===================

We have several examples to illustrate various features of the code. These
notebooks have been rendered here for the web, but the originals are
found in the ``examples/`` directory of the package, and you can run them
yourself! 

You can also try them out online directly from your browser: |binder|_
(Note: the performance of the online servers can vary widely.)


.. toctree::
    :maxdepth: 1
    :glob:

    nb/contact_map.ipynb
    nb/changing_defaults.ipynb
    nb/custom_plotting.ipynb
    nb/comparing_different_structures.ipynb
    nb/contact_trajectory.ipynb
    nb/concurrences.ipynb
    nb/exporting_data.ipynb
    nb/performance.ipynb
    nb/advanced_matplotlib.ipynb

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/dwhswenson/contact_map/{{ version }}?filepath=%2Fexamples
