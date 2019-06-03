#!/usr/bin/env python3
"""
A tool to find and remove duplicate pictures (CLI and webserver modified with permission from
@philipbl's https://github.com/philipbl/duplicate_images).

Usage:
    simages add <path> ... [--db=<db_path>] [--parallel=<num_processes>]
    simages remove <path> ... [--db=<db_path>]
    simages clear [--db=<db_path>]
    simages show [--db=<db_path>]
    simages find <path> [--print] [--delete] [--match-time] [--trash=<trash_path>] [--db=<db_path>] [--epochs=<epochs>]
    simages -h | --help
Options:
    -h, --help                Show this screen
    --db=<db_path>            The location of the database or a MongoDB URI. (default: ./db)
    --parallel=<num_processes> The number of parallel processes to run to hash the image
                               files (default: number of CPUs).
    find:
        --print               Only print duplicate files rather than displaying HTML file
        --delete              Move all found duplicate pictures to the trash. This option takes priority over --print.
        --match-time          Adds the extra constraint that duplicate images must have the
                              same capture times in order to be considered.
        --trash=<trash_path>  Where files will be put when they are deleted (default: ./Trash)
        --epochs=<epochs>     Epochs for training [default: 2]
"""

import argparse
import logging
import os
import sys
from typing import Union

import numpy as np

from .extractor import EmbeddingExtractor


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def build_parser():
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
    return parser


def parse_arguments(args):
    parser = build_parser()
    args, unknown = parser.parse_known_args(args)
    return args


def find_duplicates(
    input: Union[str, np.ndarray],
    n: int = 5,
    num_epochs: int = 2,
    num_channels: int = 3,
    show: bool = False,
    show_train: bool = False,
    show_path: bool = False,
    z_dim: int = 8,
    db = None,
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


def find_similar(db):
    extractor = EmbeddingExtractor()
    pairs, distances = extractor.duplicates()
    indices = pairs.flatten()
    import simages
    paths = [extractor.image_path(ind) for ind in indices]
    sims = simages.duplicate_images.duplicate_finder.query_paths(paths, db)



def cli():
    from docopt import docopt
    from pprint import pprint
    from simages.duplicate_images.duplicate_finder import connect_to_db,\
        add, remove, clear, show, find, delete_duplicates, display_duplicates, find_pairs

    args = docopt(__doc__)

    if args["--trash"]:
        TRASH = args["--trash"]
    else:
        TRASH = "./Trash/"

    if args["--db"]:
        DB_PATH = args["--db"]
    else:
        DB_PATH = "./db"

    if args["--parallel"]:
        NUM_PROCESSES = int(args["--parallel"])
    else:
        NUM_PROCESSES = None

    with connect_to_db(db_conn_string=DB_PATH) as db:
        if args["add"]:
            add(args["<path>"], db, NUM_PROCESSES)
        elif args["remove"]:
            remove(args["<path>"], db)
        elif args["clear"]:
            clear(db)
        elif args["show"]:
            show(db)
        elif args["find"]:
            # dups = find(db, match_time=args["--match-time"])
            dups = find_pairs(args["<path>"], db=db, epochs=int(args["--epochs"]))
            # Add similar images
            # sims = find_similar(db)
            if args["--delete"]:
                delete_duplicates(dups, db)
            elif args["--print"]:
                pprint(dups)
                print("Number of duplicates: {}".format(len(dups)))
            else:
                display_duplicates(dups, db=db)

if __name__ == "__main__":
    main()
