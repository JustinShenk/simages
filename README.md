# Simages: Find similar images :monkey_face: | :monkey_face:

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

### How to use

```python
import simages

# X is an n x m numpy array
pairs, distances = simages.
```

You can specify how many pairs you want to identify with `n`.
 
### How it works

Simages uses a convolutional autoencoder with PyTorch and compares the latent representations with [closely](https://github.com/justinshenk/closely).

### Example
```python
 
```

Output:
![example_plot](example_plot.png)
