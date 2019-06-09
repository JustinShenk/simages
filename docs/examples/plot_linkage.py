"""
Linkage
=======
simages allows visualizing the similarity of images in a dataset using
:func:`simages.linkageplot`.
"""

###############################################################################
# Ordered
# -------
# Show ordered linkage plot
#

import simages
from simages import linkageplot
import numpy as np

X = np.random.random((100, 28, 28))
embeddings = simages.EmbeddingExtractor(X, num_channels=1).embeddings

linkageplot(embeddings)

###############################################################################
# Unordered
# ---------
# Show unordered linkage plot
#

linkageplot(embeddings, ordered=False)
