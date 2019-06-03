import os
from pathlib import Path
import pytest

import matplotlib

matplotlib.use("Agg")
import numpy as np

import simages
from simages import find_duplicates, Embeddings, EmbeddingExtractor


@pytest.mark.parametrize("hw", [48, 64])
def test_find_duplicates(hw):
    data = np.random.random((100, hw, hw))
    data = data[:, np.newaxis, ...]
    embeddings = Embeddings(data, show_train=False)

    n = 3
    pairs, distances = embeddings.duplicates(n=n)
    assert len(pairs) >= n

    pairs, distances = find_duplicates(data, num_channels=1, show_train=False)
    assert len(pairs) >= n


def test_extract_embeddings():
    data = np.random.random((100, 28, 28))
    data = data[:, np.newaxis, ...]
    extractor = EmbeddingExtractor(data, num_channels=1)
    pairs, distances = extractor.duplicates(n=10)
    assert isinstance(pairs, np.ndarray)
    assert isinstance(distances, np.ndarray)

    extractor.show_images([1, 2])
    image = extractor.decode(index=5)
    assert isinstance(image, np.ndarray)

def test_conv_autoencoder():
    image_dir = os.path.join(Path(simages.__file__).parents[2], 'images','balloons')
    extractor = EmbeddingExtractor(image_dir,num_epochs=2)
    pairs, distances = extractor.duplicates(n=5)
    assert isinstance(pairs, np.ndarray)
    assert isinstance(distances, np.ndarray)
