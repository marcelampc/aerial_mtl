import rasterio
from ipdb import set_trace as st
import collections
import numpy as np
semantics = './datasets/2018IEEE_Contest/Phase2/TrainingGT/2018_IEEE_GRSS_DFC_GT_TR.tif'

# get frequency for each class
raster = rasterio.open(semantics).read()
raster = raster.flatten()
counter = collections.Counter(raster)
print(counter)
# st()
sum = 0
array = [counter[k] for k in sorted(counter.keys())]
median = np.median(array)
# for k in sorted(counter.keys()):
    # sum += counter[k] 
    # print(counter[k])

# print(sum)
weights_array = []
print('Median: {}'.format(median))
for k in sorted(counter.keys()):
    print('Key[{}]: {}'.format(k, median/counter[k]))
    weights_array.append(median/counter[k])

print(weights_array)