import os
import glob
from typing import Optional, Union

import closely
import numpy as np

from .extractor import EmbeddingExtractor


class Embeddings:
    """Create embeddings from `input` data by training an autoencoder.

    Passes arguments for `EmbeddingExtractor`.

    Attributes:
        extractor (simages.EmbeddingExtractor): workhorse for extracting embeddings from dataset
        embeddings (np.ndarray): embeddings
        pairs (np.ndarray): n closest pairs
        distances (np.ndarray): distances between n-closest pairs

    """

    def __init__(self, input: Union[np.ndarray, str], **kwargs):
        """Inits Embeddings with data."""
        if isinstance(input, str):
            if os.path.isdir(input):
                self.data_dir = input
                # Get files
                files = glob.glob(os.path.join(input, "*.*"))

                # Exclude hidden files
                files = [x for x in files if not x.startswith(".")]

                # Assume they are images
                if len(files):
                    self.embeddings = self.images_to_embeddings(self.data_dir, **kwargs)
                else:
                    raise Exception(f"Files count is {len(files)}")
        elif isinstance(input, np.ndarray):
            if input.ndim == 3 and input.shape[0] == 1:
                num_channels = 1
            elif input.ndim == 4:
                num_channels = input.shape[1]
            else:
                raise (
                    f"Data shape {input.shape} not supported, shoudld be N x C x H x W"
                )

            self.embeddings = self.array_to_embeddings(
                input, num_channels=num_channels, **kwargs
            )
        else:
            raise NotImplementedError(f"{type(input)}")

    @property
    def array(self):
        return self.extractor.embeddings

    def duplicates(self, n: int = 10):
        self.pairs, self.distances = closely.solve(self.embeddings, n=n)

        return self.pairs, self.distances

    def show_duplicates(self, n=5):
        """Convenience wrapper for `EmbeddingExtractor.show_duplicates`"""
        return self.extractor.show_duplicates(n=n)

    def images_to_embeddings(self, data_dir: str, **kwargs):
        self.extractor = EmbeddingExtractor(data_dir, **kwargs)
        return self.extractor.embeddings

    def array_to_embeddings(self, array: np.ndarray, **kwargs):
        self.extractor = EmbeddingExtractor(array, **kwargs)
        return self.extractor.embeddings

    def __repr__(self):
        return np.array_repr(self.extractor.embeddings)
