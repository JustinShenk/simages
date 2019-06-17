# :monkey: simages:monkey:
[![PyPI version](https://badge.fury.io/py/simages.svg)](https://badge.fury.io/py/simages) [![Build Status](https://travis-ci.com/justinshenk/simages.svg?branch=master)](https://travis-ci.com/justinshenk/simages)  [![Documentation Status](https://readthedocs.org/projects/simages/badge/?version=latest)](https://simages.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/188052094.svg)](https://zenodo.org/badge/latestdoi/188052094) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/justinshenk/simages/master?filepath=demo.ipynb)


Find similar images within a dataset. 

Useful for removing duplicate images from a dataset after scraping images with [google-images-download](https://github.com/hardikvasa/google-images-download).

The Python API returns `pairs, duplicates`, where pairs are the (ordered) closest pairs and distances is the 
corresponding embedding distance.

### Install

See the [installation docs](https://simages.readthedocs.io/en/latest/install.html) for all details. 

```bash
pip install simages
```

or install from source:

```bash
git clone https://github.com/justinshenk/simages
cd simages
pip install .
```

To install the interactive interface, [install mongodb](https://docs.mongodb.com/manual/installation/) and use rather `pip install "simages[all]"`.

### Demo

1. Minimal command-line interface with ```simages-show```:

![simages_demo](images/simages_demo.gif)

2. Interactive image deletion with ```simages add/find```:
![simages_web_demo](images/screenshot_server.png)

### Usage

Two interfaces exist:

1. minimal interface which plots the duplicates for visual inspection
2. mongodb + flask interface which allows interactive deletion [optional]
 
#### Minimal Interface

In your console, enter the directory with images and use `simages-show`:

```bash
$ simages-show --data-dir .
```

```
usage: simages-show [-h] [--data-dir DATA_DIR] [--show-train]
                    [--epochs EPOCHS] [--num-channels NUM_CHANNELS]
                    [--pairs PAIRS] [--zdim ZDIM] [-s]

  -h, --help            show this help message and exit
  --data-dir DATA_DIR, -d DATA_DIR
                        Folder containing image data
  --show-train, -t      Show training of embedding extractor every epoch
  --epochs EPOCHS, -e EPOCHS
                        Number of passes of dataset through model for
                        training. More is better but takes more time.
  --num-channels NUM_CHANNELS, -c NUM_CHANNELS
                        Number of channels for data (1 for grayscale, 3 for
                        color)
  --pairs PAIRS, -p PAIRS
                        Number of pairs of images to show
  --zdim ZDIM, -z ZDIM  Compression bits (bigger generally performs better but
                        takes more time)
  -s, --show            Show closest pairs

```

#### Web Interface [Optional]

Note: To install the web interface API, [install and run mongodb](https://docs.mongodb.com/manual/installation/) and use `pip install "simages[all]"` to install optional dependencies.

Add your pictures to the database (this will take some time depending on the number of pictures)

```
simages add <images_folder_path>
```

A webpage will come up with all of the similar or duplicate pictures:
```
simages find <images_folder_path>
```

```
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
```


### Python APIs

#### Numpy array

```python
from simages import find_duplicates
import numpy as np

array_data = np.random.random(100, 3, 48, 48)# N x C x H x W
pairs, distances = find_duplicates(array_data)
 
```

#### Folder

```python
from simages import find_duplicates

data_dir = "my_images_folder"
pairs, distances = find_duplicates(data_dir)
 
```

Default options for `find_duplicates` are:

```python
def find_duplicates(
    input: Union[str or np.ndarray],
    n: int = 5,
    num_epochs: int = 2,
    num_channels: int = 3,
    show: bool = False,
    show_train: bool = False,
    **kwargs
):
    """Find duplicates in dataset. Either `array` or `data_dir` must be specified.

    Args:
        input (str or np.ndarray): folder directory or N x C x H x W array
        n (int): number of closest pairs to identify
        num_epochs (int): how long to train the autoencoder (more is generally better)
        show (bool): display the closest pairs
        show_train (bool): show output every
        z_dim (int): size of compression (more is generally better, but slower)
        kwargs (dict): etc, passed to `EmbeddingExtractor`

    Returns:
        pairs (np.ndarray): indices for closest pairs of images, n x 2 array
        distances (np.ndarray): distances of each pair to each other
```

#### `Embeddings` API

```python
from simages import Embeddings
import numpy as np

N = 1000
data = np.random.random((N, 28, 28))
embeddings = Embeddings(data)

# Access the array
array = embeddings.array # N x z (compression size)

# Get 10 closest pairs of images
pairs, distances = embeddings.duplicates(n=5)

```

```python
In [0]: pairs
Out[0]: array([[912, 990], [716, 790], [907, 943], [483, 492], [806, 883]])

In [1]: distances
Out[1]: array([0.00148035, 0.00150703, 0.00158789, 0.00168699, 0.00168721])
```

#### `EmbeddingExtractor` API

```python
from simages import EmbeddingExtractor
import numpy as np

N = 1000
data = np.random.random((N, 28, 28))
extractor = EmbeddingExtractor(data, num_channels=1) # grayscale

# Show 10 closest pairs of images
pairs, distances = extractor.show_duplicates(n=10)

```

Class attributes and parameters:

```python
class EmbeddingExtractor:
    """Extract embeddings from data with models and allow visualization.

    Attributes:
        trainloader (torch loader)
        evalloader (torch loader)
        model (torch.nn.Module)
        embeddings (np.ndarray)

    """
    def __init__(
        self,
        input:Union[str, np.ndarray],
        num_channels=None,
        num_epochs=2,
        batch_size=32,
        show_train=True,
        show=False,
        z_dim=8,
        **kwargs,
    ):
    """Inits EmbeddingExtractor with input, either `str` or `np.nd.array`, performs training and validation.
    
    Args:
    input (np.ndarray or str): data
    num_channels (int): grayscale = 1, color = 3
    num_epochs (int): more is better (generally)
    batch_size (int): number of images per batch
    show_train (bool): show intermediate training results
    show (bool): show closest pairs
    z_dim (int): compression size
    kwargs (dict)
    
    """

```

Specify tne number of pairs to identify with the parameter `n`.
 
### How it works

*simages* uses a convolutional autoencoder with PyTorch and compares the latent representations with [closely](https://github.com/justinshenk/closely) :triangular_ruler:.

#### Dependencies

*simages* depends on
the following packages:

- [closely](https://github.com/justinshenk/closely)
- [torch](https://pytorch.org)
- [torchvision](https://pytorch.org)
- scikit-learn
- matplotlib

The following dependencies are required for the interactive deleting interface:
 
- pymongodb
- fastcluster
- flask
- jinja2
- dnspython
- python-magic
- termcolor

### Cite

If you use simages, please cite it:
```
    @misc{justin_shenk_2019_3237830,
      author       = {Justin Shenk},
      title        = {justinshenk/simages: v19.0.1},
      month        = jun,
      year         = 2019,
      doi          = {10.5281/zenodo.3237830},
      url          = {https://doi.org/10.5281/zenodo.3237830}
    }
```
