import os
import sys

from torch.utils.data.dataset import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader, has_file_allowed_extension


class PILDataset(Dataset):
    """PIL dataset."""

    def __init__(self, pil_list, transform=None):
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


class SingleFolderDataset(VisionDataset):
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
        root,
        loader=default_loader,
        extensions=None,
        transform=None,
        is_valid_file=None,
    ):
        super(SingleFolderDataset, self).__init__(root)
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

    def __getitem__(self, index):
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


def make_dataset_wo_targets(dir, extensions=None, is_valid_file=None):
    """Modified from torchvision `make_dataset`."""
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:

        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    for fname in sorted(os.listdir(dir)):
        path = os.path.join(dir, fname)
        if is_valid_file(path):
            item = path
            images.append(item)

    return images
