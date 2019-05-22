#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

import os
import re

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with open(os.path.join(here, *parts), "r", encoding="utf8") as fp:
        return fp.read()


# Get package version
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


requirements = ["numpy", "scikit-learn", "torch", "torchvision", "Pillow", "closest_pairs"]

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="similar_images",
    version=find_version("similar_images", "__init__.py"),
    description="Find similar images (duplicates) in a dataset",
    url="https://github.com/justinshenk/similar-images",
    author="Justin Shenk",
    author_email="shenkjustin@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">= 3.6",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    keywords="computer vision images dataset preprocessing duplicates classification embeddings",
    zip_safe=False,
)
