import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from simages import find_duplicates, Embeddings


@pytest.mark.parametrize("hw",[48,64,128])
def test_import(hw):
    data = np.random.random((100, hw, hw))
    data = data[:, np.newaxis, ...]
    embeddings = Embeddings(data, show_train=False)
    pairs, distances = embeddings.duplicates(n=3)
    # find_duplicates(data, show_train=False)
