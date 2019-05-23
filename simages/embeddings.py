import os
import glob
from typing import Union

import closely
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from .extractor import EmbeddingExtractor


class Embeddings():
    """Create embeddings from `input` data."""
    def __init__(self, input:Union[np.ndarray, str], **kwargs):
        if isinstance(input, str):
            if os.path.isdir(input):
                self.data_dir = input
                # Get files
                files = glob.glob(os.path.join(input, "*.*"))

                # Exclude hidden files
                files = [x for x in files if not x.startswith('.')]

                # Assume they are images
                if len(files):
                    self.embeddings = self.images_to_embeddings(self.data_dir, **kwargs)
                else:
                    raise Exception(f"Files count is {len(files)}")
        elif isinstance(input, np.ndarray):
            self.embeddings = self.array_to_embeddings(input, **kwargs)
        else:
            raise NotImplementedError(f"{type(input)}")

    @property
    def array(self):
        return self.extractor.embeddings

    def duplicates(self, n:int=5):
        assert isinstance(self.embeddings, np.ndarray)
        self.pairs, self.distances = closely.solve(self.embeddings, n=n)

        return self.pairs, self.distances

    def images_to_embeddings(self, data_dir:str, **kwargs):
        self.extractor = EmbeddingExtractor(data_dir=data_dir, **kwargs)
        return self.extractor.embeddings

    def array_to_embeddings(self, array:np.ndarray, **kwargs):
        self.extractor = EmbeddingExtractor(array=array, **kwargs)
        return self.extractor.embeddings

    def __repr__(self):
        return np.array_repr(self.extractor.embeddings)

    def show(self, img, title=""):
        if isinstance(img, torch.Tensor):
            npimg = img.numpy()
        else:
            raise NotImplementedError(f"{type(img)}")

        plt.subplots()
        plt.title(title)
        plt.imshow(np.transpose(npimg,(1,2,0)).squeeze(), interpolation='nearest')
        plt.show()

    def show_images(self, indices, title=""):
        tensors = [self.extractor.dataloader.dataset[idx].cpu() for idx in indices]
        self.show(torchvision.utils.make_grid(tensors), title=title)

    def show_duplicates(self, n=5):
        if not hasattr(self, "pairs"):
            self.pairs, self.distances = self.duplicates(n=n)
        elif len(self.pairs) < n:
            print(f"Requested duplicates {n} is greater than {len(self.pairs)}, recalculating...")
            self.pairs, self.distances = self.duplicates(n=n)

        # Plot pairs
        imgs = []
        for pair in self.pairs:
            img0 = self.extractor.dataloader.dataset[pair[0]].cpu()
            img1 = self.extractor.dataloader.dataset[pair[1]].cpu()
            self.show(torchvision.utils.make_grid([img0, img1]), title=pair)