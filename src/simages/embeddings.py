import os
import glob
from typing import Optional, Union

import closely
import numpy as np

from .extractor import EmbeddingExtractor


class Embeddings:
    """Create embeddings from `input` data."""

    def __init__(self, input: Union[np.ndarray, str], **kwargs):
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
            if input.ndim == 3:
                num_channels = 1
            elif input.ndim == 4:
                num_channels = input.shape[1]

            self.embeddings = self.array_to_embeddings(input, num_channels = num_channels, **kwargs)
        else:
            raise NotImplementedError(f"{type(input)}")

    @property
    def array(self):
        return self.extractor.embeddings

    def duplicates(self, n: Optional[int] = None):
        self.pairs, self.distances = closely.solve(self.embeddings, n=n)

        return self.pairs, self.distances

    def images_to_embeddings(self, data_dir: str, **kwargs):
        self.extractor = EmbeddingExtractor(data_dir=data_dir, **kwargs)
        return self.extractor.embeddings

    def array_to_embeddings(self, array: np.ndarray, **kwargs):
        print("kwargs", kwargs)
        self.extractor = EmbeddingExtractor(array=array, **kwargs)
        return self.extractor.embeddings

    def __repr__(self):
        return np.array_repr(self.extractor.embeddings)