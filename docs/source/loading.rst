Loading Data
============

Data is loaded using the :class:`~simages.extractor.EmbeddingExtractor` class.

``EmbeddingExtractor`` is used to extract embeddings by

- Train an autoencoder on the images
- Identify similar images from the embeddings of the autoencoder
- Plot and visualize the results

Dataset can be provided as a numpy array or as an image folder path.

.. autoclass:: simages.extractor.EmbeddingExtractor


**Numpy Array**

Load data with::

    from simages import EmbeddingExtractor
    import numpy as np

    # Create grayscale (1-channel) samples
    X = np.random.random((100,28,28))
    extractor =  EmbeddingExtractor(X, num_channels=1)

    # Find duplicates
    pairs, distances = extractor.find_duplicates()


**Image Folder**::

    from simages import EmbeddingExtractor

    # Point to Folder
    data_dir = "downloads"
    extractor =  EmbeddingExtractor(data_dir)

    # Find duplicates
    pairs, distances = extractor.find_duplicates()

    # Show duplicates
    extractor.show_duplicates(n=5)

Duplicates can be identified using the ``simages`` command::

    $ simages add `{image_folder}`

    $ simages find `{image_folder}`

Duplicates can be deleted on the webserver as described at :doc:`removing`.