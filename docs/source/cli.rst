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


Add your pictures to the database
(this will take some time depending on the number of pictures)

.. code-block::bash

    simages add <images_folder_path>


A webpage will come up with all of the similar or duplicate pictures


.. code-block::bash

    simages find <images_folder_path>


    Usage:
        simages add <path> ... [--db=<db_path>] [--parallel=<num_processes>]
        simages remove <path> ... [--db=<db_path>]
        simages clear [--db=<db_path>]
        simages show [--db=<db_path>]
        simages find <path> [--print] [--delete] [--match-time] [--trash=<trash_path>] [--db=<db_path>] [--epochs=<epochs>]
        simages -h | --help
    Options:
        -h, --help                Show this screen
        --db=<db_path>            The location of the database or a MongoDB URI. (default: ./db)
        --parallel=<num_processes> The number of parallel processes to run to hash the image
                                   files (default: number of CPUs).
        find:
            --print               Only print duplicate files rather than displaying HTML file
            --delete              Move all found duplicate pictures to the trash. This option takes priority over --print.
            --match-time          Adds the extra constraint that duplicate images must have the
                                  same capture times in order to be considered.
            --trash=<trash_path>  Where files will be put when they are deleted (default: ./Trash)
            --epochs=<epochs>     Epochs for training [default: 2]


``simages`` calls :func:`~simages.main.cli`.