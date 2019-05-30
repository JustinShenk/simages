Running simages from the console
================================

 .. _my-reference-label:

simages can be run locally in the terminal with ``simages-show``.

Usage::

    simages-show --data-dir .


See all the options for ``simages-show`` with ``simages-show --help``:

.. argparse::
   :ref: simages.main.build_parser
   :prog: simages

``simages-show`` calls :func:`~simages.main.find_duplicates`:


.. autofunction:: simages.main.find_duplicates
    :members:
    :undoc-members:
    :show-inheritance:
