import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from simages import find_duplicates, Embeddings


@pytest.mark.parametrize("hw", [48, 64, 128])
def test_find_duplicates(hw):
    data = np.random.random((100, hw, hw))
    data = data[:, np.newaxis, ...]
    embeddings = Embeddings(data, show_train=False)
    n = 3
    pairs, distances = embeddings.duplicates(n=n)
    assert len(pairs) >= n

    pairs, distances = find_duplicates(data, show_train=False)
    assert len(pairs) >= n
