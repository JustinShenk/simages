#!/usr/bin/env python3

import argparse
import sys
from typing import Optional

import numpy as np

from simages import Embeddings, EmbeddingExtractor


def parse_arguments(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        default=None,
        help="Folder containing image data"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively gather data from folders in `data_dir`"
    )
    parser.add_argument(
        "--show_train",
        action="store_true",
        default=None,
        help="Show training of embedding extractor every epoch"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=2,
        help="Number of passes of dataset through model for training. More is better but takes more time."
    )
    parser.add_argument(
        "-c",
        "--num_channels",
        type=int,
        default=3,
        help="Number of channels for data (1 for grayscale, 3 for color)"
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=10,
        help="Number of pairs of images to show"
    )
    parser.add_argument(
        "-z",
        "--zdim",
        type=int,
        default=8,
        help="Compression bits (bigger generally performs better but takes more time)"
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        default=False,
        help="Show closest pairs"
    )
    args, unknown = parser.parse_known_args(args)
    return args


def find_duplicates(array:Optional[np.ndarray]=None, data_dir:Optional[str]=None, n: int = 5, num_epochs:int=2, num_channels:int=3, show:bool=False, show_train:bool=False, **kwargs):
    """Find duplicates in dataset.
    Args:
        array (np.ndarray, optional)
        data_dir (str, optional)
        n (int)
        num_epochs (int)
        show (bool)
        show_train (bool)
        z_dim (int)
        kwargs (dict)

    Returns:
        pairs (list)
        distances (np.ndarray)

    """
    if array is not None:
        extractor = EmbeddingExtractor(array=array, num_epochs=num_epochs, num_channels=num_channels, show_train=show_train, show=show, **kwargs)
    elif data_dir is not None:
        extractor = EmbeddingExtractor(data_dir=data_dir, num_epochs=num_epochs, num_channels=num_channels, show_train=show_train, show=show, **kwargs)

    if show:
        pairs, distances = extractor.show_duplicates(n=n)
    else:
        pairs, distances = extractor.duplicates(n=n)
    return pairs, distances

def main():
    args = parse_arguments(sys.argv[1:])

    find_duplicates(data_dir=args.data_dir, n=args.pairs, num_epochs=args.epochs, num_channels=args.num_channels, show=args.show, show_train=args.show_train)

if __name__ == '__main__':
    main()