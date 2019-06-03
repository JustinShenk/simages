import glob
import os
from typing import Callable, Optional

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension


class PILDataset(Dataset):
    """PIL dataset."""

    def __init__(self, pil_list: list, transform: Optional[Callable] = None):
        """
        Args:
            pil_list (list of PIL images)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pil_list = pil_list
        self.transform = transform

    def __len__(self):
        return len(self.pil_list)

    def __getitem__(self, idx):
        sample = self.pil_list[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImageFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/xxx.ext
        root/xxy.ext
        root/xxz.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        loader: Callable = default_loader,
        extensions: Optional[list] = None,
        transform: Optional[list] = None,
        is_valid_file: Optional[Callable] = None,
    ):
        super(ImageFolder, self).__init__(root)
        self.transform = transform

        samples = make_dataset_wo_targets(self.root, extensions, is_valid_file)
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n"
                    "Supported extensions are: " + ",".join(extensions)
                )
            )

        self.loader = loader
        self.extensions = extensions

        self.samples = samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)


def make_dataset_wo_targets(
    dir: str,
    extensions: Optional[list] = None,
    is_valid_file: Optional[Callable] = None,
):
    """Modified from torchvision's `make_dataset`."""
    images = []
    dir = os.path.expanduser(dir)

    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    files = [f for f in glob.glob(os.path.join(dir, "**/*.*"), recursive=True)]
    for fname in sorted(files):
        path = os.path.join(dir, fname)
        if is_valid_file(path):
            item = path
            images.append(item)

    return images


class DatasetDB(Dataset):
    def __init__(self, db_name='images', col_name='eval', transform=None):
        self._label_dtype = np.int32
        self.transform = transform

        from pymongo import MongoClient
        client = MongoClient('localhost', 27017)
        db = client[db_name]
        self.col = db[col_name]
        self.examples = list(self.col.find({}, {'imgs': 0}))

    def __len__(self):
        return len(self.examples)

    def pil_loader(self, f):
        from PIL import Image
        import io
        with Image.open(io.BytesIO(f)) as img:
            return img.convert('RGB')

    def __getitem__(self, i):
        _id = self.examples[i]['_id']
        doc = self.col.find_one({'_id': _id})

        img = doc['imgs'][0]['picture']
        img = self.pil_loader(img)

        if self.transform:
            img = self.transform(img)

        return img, _id