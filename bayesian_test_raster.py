# simplified main
from options.extra_args_dfc import MTL_Raster_Options
from dataloader.data_loader import CreateDataLoader
from models.getmodels import create_model
from ipdb import set_trace as st

# Load options
opt = MTL_Raster_Options().parse()

from models.test_model_raster import TestModel
model = TestModel()
model.initialize(opt)
print("model [%s] was created" % (model.name()))
# model.test_raster()

import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# todo: add dropout every layer of the network
sum_outputs = []

images_list_ = []
n_iters = 30

# create folders
model.create_save_folders(subfolders=['bayesian'])

# create data loader and start setings
model.initialize_test_bayesian(opt)

# get data_loader size to run test_bayesian by index
data_loader_size = model.get_data_loader_size()

for i in tqdm(range(data_loader_size)): # for each image in the data loader

    # get error and output list to calculate mean
    error_list, out_list, target_np = model.test_bayesian(i, n_iters)
    error_np = np.array(error_list).mean(axis=0)
    out_np = np.array(out_list)

    # get mean image and var
    mean_out_np = out_np.mean(axis=0)
    var = np.square(out_np - mean_out_np).mean(axis=0)
    mean_error = (np.abs(out_np - target_np)).mean(axis=0)
    
    # save images
    meta_data = model.get_meta_data()
    shape = model.get_shape()
    model.save_dsm_as_raster(mean_out_np, 'bayesian/mean_image_{}.tif'.format(i+20), meta_data, shape)
    model.save_dsm_as_raster(var, 'bayesian/var_image_{}.tif'.format(i+20), meta_data, shape)
    model.save_dsm_as_raster(mean_error, 'bayesian/m_error_image_{}.tif'.format(i+20), meta_data, shape)

    print(mean_error.mean())


model.save_merged_rasters('bayesian', 'mean')
model.save_merged_rasters('bayesian', 'var')
model.save_merged_rasters('bayesian', 'm_error')

