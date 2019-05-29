# Simages :monkey_face: :monkey_face:
[![PyPI version](https://badge.fury.io/py/simages.svg)](https://badge.fury.io/py/simages) [![Build Status](https://travis-ci.com/justinshenk/simages.svg?branch=master)](https://travis-ci.com/justinshenk/simages) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/justinshenk/simages/master?filepath=demo.ipynb)


Find similar images within a dataset. Useful for finding duplicates images.

### Getting Started

```bash
pip install simages
```

or install from source:
```bash
git clone https://github.com/justinshenk/simages
cd simages
pip install .
```

### Demo

![simages_demo](images/simages_demo.gif)

### Command Line

In your console, enter the directory with images and use `simages-show`:

```bash
$ simages-show --data-dir .
```

Usage:

```
usage: simages-show [-h] [--data-dir DATA_DIR] [--recursive] [--show-train]
                    [--epochs EPOCHS] [--num-channels NUM_CHANNELS]
                    [--pairs PAIRS] [--zdim ZDIM] [-s]
```

Optional arguments:
```
  -h, --help            show this help message and exit
  --data-dir DATA_DIR, -d DATA_DIR
                        Folder containing image data
  --recursive, -r       Recursively gather data from folders in `data_dir`
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

### Find Duplicates

#### Numpy array

```python
from simages import find_duplicates

array_data # N x C x H x W
find_duplicates(array_data)
 
```

#### Folder

```python
from simages import find_duplicates

data_dir = "my_images_folder"
find_duplicates(data_dir)
 
```

Default options for `find_duplicates` are:

```
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

# X is an n x m numpy array
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

# X is an n x m numpy array
N = 1000
data = np.random.random((N, 28, 28))
extractor = EmbeddingExtractor(data, num_channels=1)

# Show 10 closest pairs of images
pairs, distances = extractor.show_duplicates(n=10)

```

Class attributes and parameters:

```python
class EmbeddingExtractor:
    """Extract embeddings from data with models and allow visualization.

    Attributes:
        trainloader (torch loader)
        evalloader (torch loader
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

You can specify how many pairs you want to identify with `n`.
 
### How it works

Simages uses a convolutional autoencoder with PyTorch and compares the latent representations with [closely](https://github.com/justinshenk/closely).
