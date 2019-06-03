.. simages documentation master file, created by
   sphinx-quickstart on Mon Jan 28 23:36:32 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

simages |version|
=================

🐵 Similar image detection in Python 🐵

simages allows detecting similar images in an image folder or numpy array.

Description
-----------

Detect similar images (eg, duplicates) in an image folder. Behind the curtain, simages uses a PyTorch autoencoder to
train embeddings. The embeddings are compared with each other to create a distance matrix. The closest pairs of
images are then presented on screen.

.. only:: html

   .. figure:: https://raw.githubusercontent.com/justinshenk/simages/master/images/simages_demo.gif

      Demo of visualizing training (``simages --show-train`` option) and found duplicates with the :ref:`simages-show command <cli>`.


If you use simages in your publications, please cite "`simages: Similar image detection with Python. https://github.com/justinshenk/simages <https://github.com/justinshenk/simages>`_."


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Installation <install>
   Examples Gallery <gallery/index>
   Command Line <cli>

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Loading Data <loading>
   Building Embeddings <build>
   Removing Duplicates <removing>

.. toctree::
   :maxdepth: 1
   :caption: Reference Guide

   Reference to All Attributes and Methods <reference>

.. toctree::
  :maxdepth: 1
  :caption: Developer

  Contributing to simages <contributing>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
