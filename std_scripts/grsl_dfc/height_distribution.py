import rasterio
from os.path import join
import glob
from ipdb import set_trace as st
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib2tikz import save as tikz_save

def plot(flat_depths, title, filename):
    # plt.figure()
    ax = plt.axes()
    # sns.distplot(flat_depths, ax=ax, kde_kws={"label": label})
    ax.set_title(title)
    root = 'std_scripts/grss_dfc/plots/'
    if not os.path.isdir(root):
        os.makedirs(root)
    sns.distplot(flat_depths, ax=ax)
    # plt.show()
    # plt.savefig(join(root, filename + '.png'), transparent=True)
    # tikz_save(join(root, filename + '.tex'))

    # sns.distplot(flat_depths, ax=ax, rug=True, hist=False)
    # plt.savefig(join(root, filename + '_2.png'), transparent=True)

root = './datasets/2018IEEE_Contest/Phase2/LidarGeoTiffRasters'
paths = [glob.glob(join(root, 'DSM_C12/*.tif'))[0], glob.glob(join(root, 'DEM_C123_TLI/*.tif'))[0]]

delta = 0.1

raster_np_1 = rasterio.open(paths[0]).read()
raster_np_2 = rasterio.open(paths[1]).read()

# get rmse from 

mask_raster = ((raster_np_1 < 9000) * (raster_np_2 < 9000)).flatten()

height_flatten = mask_raster * (raster_np_1.flatten() - raster_np_2.flatten())
# x, y = np.histogram(height_flatten, bins=20, density=True)

# plot(height_flatten, 'Height Distribution', 'height_distribution')
# normalize between 0-1
min_v, max_v = height_flatten.min(), height_flatten.max()
height_normalized = (height_flatten - min_v) / (max_v - min_v)
# plot(height_normalized, 'Normalized Height Distribution', 'height_distribution')

# standardization:
std_ = np.std(height_flatten)
mean_ = np.mean(height_flatten)

height_std = (height_flatten - mean_) / std_
min_v, max_v = height_std.min(), height_std.max()
height_std = (height_std - min_v) / (max_v - min_v)
plot(height_std, 'Std Height Distribution', 'height_distribution')

# st()
plt.show()