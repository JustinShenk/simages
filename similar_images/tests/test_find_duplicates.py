import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from similar_images import find_duplicates

@pytest.mark.parametrize("hw",[48,64,128])
def test_import(hw):
    data = np.random.random((100, hw, hw))
    data = data[:, np.newaxis, ...]
    find_duplicates(data, show_train=False)
