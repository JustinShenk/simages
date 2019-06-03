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

Alternatively, removing duplicate images in a dataset interactively is easy with ``simages``.

- [Install mongodb](https://docs.mongodb.com/manual/installation/) on your system.

- Add images to the database via ``simages add {image_folder_path}``.

- Find duplicates and run the web server with ``simages find``.




