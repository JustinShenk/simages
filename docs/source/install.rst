Installation
============

Installing simages is pretty simple.

If you haven't got it, obtain Python_ (version 3.6 or greater).

.. _Python: https://www.python.org/

Install with pip::

   pip install simages


If you wish to install the latest development version, clone the GitHub_ repository and use the setup script::

   git clone https://github.com/justinshenk/simages.git
   cd simages
   pip install .


Next you can go to :doc:`the Command Line Interface <cli>`.

Dependencies
------------

Installation with pip should also include all dependencies, but a complete list is

- numpy_
- matplotlib_
- closely_
- torch_
- torchvision_
- pymongodb
- fastcluster
- flask
- jinja2
- dnspython
- python-magic
- termcolor

The optional web and :ref:`simages command line interface <cli>` requires installing mongodb::

    sudo apt -y install mongodb-server
    # Disable Mongo service autostart
    sudo systemctl disable mongodb.service
    # Stop Mongo service
    sudo service mongodb stop


.. _GitHub: https://github.com/justinshenk/simages

.. _numpy: https://www.numpy.org

.. _closely: https://github.com/justinshenk/closely

.. _matplotlib: https://matplotlib.org

.. _torch: https://pytorch.org

.. _torchvision: https://pytorch.org/docs/stable/torchvision
