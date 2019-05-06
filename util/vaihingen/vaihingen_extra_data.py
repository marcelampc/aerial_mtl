# Extra information about nyuv2
import numpy as np

# Number of classes:
n_classes = 7

# set 0 to first label
# weights=np.ones(1, n_classes)
weights = [
            0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
colors = [  
        [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]
        ]