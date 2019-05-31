"""
Comparing
---------
simages allows comparing trajectories using various methods.
"""
import simages
import numpy as np

X = np.random.random((100, 28, 28))
simages.find_duplicates(X, num_channels=1, show=True)
