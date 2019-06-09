Building Embeddings
===================

Embeddings are extracted from the images by training a convolutional autoencoder.

The models available are listed in :mod:`~simages.models`. The default autoencoder
is a 3-layer convolutional autoencoder, `~simages.BasicAutoencoder::

    from simages import EmbeddingExtractor

    extractor = EmbeddingExtractor(image_dir)


``extractor`` allows identifying images corresponding to the embeddings.

For example, if ``extractor.duplicates()`` returns ``pairs`` ``[2, 3]``, the images
corresponding to embeddings 2 and 3 can be viewed with :meth:`simages.extractor.EmbeddingExtractor.image_paths`:

.. automethod:: simages.extractor.EmbeddingExtractor.image_paths



After building the embeddings, the embeddings can be extracted with::

    emeddings = extractor.embeddings

Extraction is performed by :class:`~simages.extractor.EmbeddingExtractor`:

.. autoclass:: simages.extractor.EmbeddingExtractor
    :members:
    :undoc-members:
    :show-inheritance:


