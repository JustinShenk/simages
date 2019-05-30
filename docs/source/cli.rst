Command Line
============

simages can be run locally via the terminal with ``simages-show``.

Usage::

    simages-show --data-dir .


Get help with ``simages-show --help``:

.. argparse::
   :ref: simages.main.build_parser
   :prog: simages

``simages-show`` calls :func:`~simages.main.find_duplicates`:


.. autofunction:: simages.main.find_duplicates
    :members:
    :undoc-members:
    :show-inheritance:
