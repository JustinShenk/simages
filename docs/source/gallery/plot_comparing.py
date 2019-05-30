"""
Comparing
---------
simages allows comparing trajectories using various methods

# .. code::
#
#     import simages
#     import numpy as np
#
#     data = np.load("../images/data.npy")
#     simages.find_duplicates(data, num_channels=1, show=True)

"""
import simages
import numpy as np

data = np.load("../images/data.npy")
simages.find_duplicates(data, num_channels=1, show=True)

plt.show()
