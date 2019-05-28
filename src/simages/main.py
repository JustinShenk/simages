import numpy as np

from simages import Embeddings


def find_duplicates(array: np.ndarray, n: int = 5, show_train=True, **kwargs):
    embeddings = Embeddings(array, show_train=show_train, **kwargs)
    pairs, distances = embeddings.duplicates(n=n)
    return pairs, distances
