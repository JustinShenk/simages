#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from typing import Optional, Union

import numpy as np

from simages import Embeddings, EmbeddingExtractor


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def parse_arguments(args):
    parser = argparse.ArgumentParser(
        description="Find similar pairs of images in a folder"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        action="store",
        type=str,
        default=None,
        help="Folder containing image data",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        default=False,
        help="Recursively gather data from folders in `data_dir`",
    )
    parser.add_argument(
        "--show-train",
        "-t",
        action="store_true",
        default=None,
        help="Show training of embedding extractor every epoch",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        action="store",
        type=int,
        default=2,
        help="Number of passes of dataset through model for training. More is better but takes more time.",
    )
    parser.add_argument(
        "--num-channels",
        "-c",
        action="store",
        type=int,
        default=3,
        help="Number of channels for data (1 for grayscale, 3 for color)",
    )
    parser.add_argument(
        "--pairs",
        "-p",
        action="store",
        type=int,
        default=10,
        help="Number of pairs of images to show",
    )
    parser.add_argument(
        "--zdim",
        "-z",
        action="store",
        type=int,
        default=8,
        help="Compression bits (bigger generally performs better but takes more time)",
    )
    parser.add_argument(
        "--image-path", "-i", action="store_true", help="Show image paths of duplicates"
    )
    args, unknown = parser.parse_known_args(args)
    return args


def find_duplicates(
    input: Union[str, np.ndarray] = None,
    n: int = 5,
    num_epochs: int = 2,
    num_channels: int = 3,
    show: bool = False,
    show_train: bool = False,
    show_path: bool = False,
    z_dim: int = 8,
    **kwargs
):
    """Find duplicates in dataset. Either `array` or `data_dir` must be specified.

    Args:
        input (str or np.ndarray): folder directory or N x C x H x W array
        n (int): number of closest pairs to identify
        num_epochs (int): how long to train the autoencoder (more is generally better)
        show (bool): display the closest pairs
        show_train (bool): show output every
        show_path (bool): show image paths of duplicates instead of index
        z_dim (int): size of compression (more is generally better, but slower)
        kwargs (dict): etc, passed to `EmbeddingExtractor`

    Returns:
        pairs (np.ndarray): indices for closest pairs of images
        distances (np.ndarray): distances of each pair to each other

    """
    if isinstance(input, np.ndarray):
        extractor = EmbeddingExtractor(
            input=input,
            num_epochs=num_epochs,
            num_channels=num_channels,
            show=show,
            show_train=show_train,
            z_dim=z_dim,
            **kwargs
        )
    elif isinstance(input, str):
        extractor = EmbeddingExtractor(
            input=input,
            num_epochs=num_epochs,
            num_channels=num_channels,
            show=show,
            show_train=show_train,
            show_path=show_path,
            z_dim=z_dim,
            **kwargs
        )

    if show:
        pairs, distances = extractor.show_duplicates(n=n)
    else:
        pairs, distances = extractor.duplicates(n=n)
    return pairs, distances


def main():
    """Main entry point for `simages-show` via command line."""
    args = parse_arguments(sys.argv[1:])

    find_duplicates(
        input=args.data_dir,
        n=args.pairs,
        num_epochs=args.epochs,
        num_channels=args.num_channels,
        show=True,
        show_train=args.show_train,
        show_path=args.image_path,
    )


if __name__ == "__main__":
    main()
