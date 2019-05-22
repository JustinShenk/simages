import numpy as np
import pytest

from similar_images import Embeddings, EmbeddingExtractor, find_duplicates

@pytest.mark.parametrize("hw",[48,64,128])
def test_import(hw):
    data = np.random.random((1000, hw, hw))
    data = data[:, np.newaxis, ...]
    find_duplicates(data)