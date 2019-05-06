# based on https://github.com/pytorch/examples/blob/master/super_resolution/
# dataset.py
# Instead of applying data augmentation to PIL.Image, apply to numpy (from AB and NA)
# custom dataset to store 2 images

# Expects that files are like:
# |- main_folder
# |- |- train
# |- |- |- gt
# |- |- |- label
# |- |- test
# |- |- |- gt
# |- |- |- label
# |- |- val
# |- |- |- gt
# |- |- |- label

# from IPython import get_ipython
# ipython = get_ipython()

# if '__IPYTHON__' in globals():
#     ipython.magic('load_ext autoreload')
#     ipython.magic('autoreload 2')

import torch.utils.data as data

import os
from os import listdir
from os.path import join
from PIL import Image
from ipdb import set_trace as st
import random
from math import pow
import numpy as np
from tqdm import tqdm
from random import randint

import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .online_data_augmentation import DataAugmentation

# global variable to apply crop
state = 0

def get_paths_list(opt, data_split, phase, dataset_name='dfc'):
    if dataset_name == 'dfc':
        from .dataset_bank import dataset_dfc
        return dataset_dfc(opt.dataroot, data_split, phase, model=opt.model, which_raster=opt.which_raster)
    elif dataset_name == 'vaihingen':
        from .dataset_bank import dataset_vaihingen
        return dataset_vaihingen(opt.dataroot, data_split, phase, model=opt.model)

def get_random_pos(img, window_shape):
    """ Extract of 2D random patch of shape window_shape in the image """
    h, w = window_shape
    H, W = img.shape[-2:]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2

def sliding_window(image, step=10, window_size=(20,20)):
    """ Slide a window_shape window across the image with a stride of step """
    step = step if len(step) == 2 else step * 2 
    window_size = window_size if len(window_size) == 2 else window_size * 2 
    height, width = image.shape if len(image.shape) == 2 else (image.shape[1], image.shape[2])
    for x in range(0, width, step[0]):
        if x + window_size[0] >= width:
            x = width - window_size[0]
        for y in range(0, height, step[1]):
            if y + window_size[1] >= height:
                y = height - window_size[1]
            yield x, x + window_size[0], y, y + window_size[1]

def str2bool(values):
    return [v.lower() in ("true", "t") for v in values]

def normalize_raster_to_numpy(data):
    return np.asarray(((data.transpose(1,2,0) + 1) / 2.0) * 255.0).astype('uint8')

def numpy_to_normalized_raster(data):
    return (((data.astype('float32') / 255.0) * 2.0) - 1.0).transpose(2,0,1)

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def resize_rgb(data, image_size, mode):
    img = normalize_raster_to_numpy(data)
    img = resize(img, image_size, mode)
    img = numpy_to_normalized_raster(img)
    return img

def normalized_rgb_to_pil_image(data):
    return Image.fromarray(normalize_raster_to_numpy(data))

def resize(data, image_size, mode):
    return np.array(Image.fromarray(data).resize(image_size, mode))

def load_rgb_and_label(input_list, phase, target_path=None, dfc_preprocessing=0):
    """
    load all depths/labels to label_cache with crop
    """
    # ToDo: fix this mess of having two almost similar functions... 
    import rasterio
    from rasterio.mask import mask
    from rasterio.plot import reshape_as_image
    import geopandas as gpd
    from shapely.geometry import box
    from scipy.misc import imresize
    
    if phase != 'test':
        raster_target = [rasterio.open(path) for path in target_path]
    
    print('Loading {} patches...'.format(phase))
    for i, img_input_path in enumerate(tqdm(input_list)):
        with rasterio.open(img_input_path) as raster_rgb:

            # normalize data between -1 and 1 and append to cache
            raster_rgb_numpy = raster_rgb.read()
            raster_rgb_norm = ((raster_rgb_numpy.astype('float32') / 255.0) * 2.0) - 1.0
            pil_shape = (raster_rgb_norm.shape[-2:])[::-1]
            
            if dfc_preprocessing == 2: # make input same resolution as output
                pil_shape = [dim // 10 for dim in pil_shape] # Resize
                raster_rgb_norm = resize_rgb(raster_rgb_norm, pil_shape, Image.BILINEAR)

            if phase == 'test':
                yield raster_rgb_norm #, pil_depth_patch_shape # To add labels for test
            else:
                labels_patches = []

                # get bounds and crop from depth_image_raster
                bounds = raster_rgb.bounds
                bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0])
                coords = getFeatures(geo)
                labels_patches_masks = [mask(image_raster, shapes=coords, crop=True) for image_raster in raster_target]

                sem_label = labels_patches_masks[0][0]
            
                if dfc_preprocessing == 0:
                    sem_label = [resize(sem_label[0], pil_shape, Image.NEAREST)]
                elif dfc_preprocessing == 1:
                    pil_shape = (raster_rgb_norm.shape[-2:])[::-1]
                    pil_shape = [dim // 2 for dim in pil_shape] # Resize
                    raster_rgb_norm = resize_rgb(raster_rgb_norm, pil_shape, Image.BILINEAR)
                
                # labels_patches.append(sem_label)
                yield raster_rgb_norm, sem_label

def get_min_max(raster, which_raster):
    if which_raster != 'dsm':
        dsm, dem = raster[0].read(), raster[1].read()
        mask_dsm = (dsm < 9000) * (dem < 9000)
        height = mask_dsm * (dsm - dem)
        return height.max(), height.min()
    else:
        raster_np = mask_invalid_depth(raster[0].read())
        return raster_np.max(), raster_np.min()

def load_labels_raster(*path_list):
    import rasterio
    from rasterio.profiles import DefaultGTiffProfile
    target_np = []
    for path in path_list:
        # target = rasterio.open(path).read()
        target = np.array(Image.open(path))
        if target.dtype == 'uint8' and len(target.shape) == 3:
            from util.util import colors_to_labels
            colors_isprs = [ 
                            [0, 0, 0],
                            [255, 255, 255],
                            [0, 0, 255],
                            [0, 255, 255],
                            [0, 255, 0],
                            [255, 255, 0],
                            [255, 0, 0],
                            ]
            # target = colors_to_labels(target.transpose(1,2,0), colors_isprs)
            target = colors_to_labels(target, colors_isprs)
        else: 
            if target.dtype == 'uint8' and target.max() > 8: # height
                # values between 0 and 1
                target = target.astype(np.float32) / 255
        target_np.append(target)
    return target_np

# def load_raster(input_list, target_list):
#     import rasterio
#     image_list = []
#     for path_input, path_output in tqdm(zip(input_list, target_list)):
#         raster_rgb_numpy = rasterio.open(path_input).read()
#         raster_rgb_norm = ((raster_rgb_numpy.astype('float32') / 255.0) * 2.0) - 1.0
#         yield raster_rgb_norm, rasterio.open(path_output).read()

def is_file(*filenames):
    for f in filenames:
        if not os.path.isfile(f):
            raise KeyError('{} is not a file !'.format(f))

def load_raster(input_list, target_list, phase='train'):
    import rasterio, os
    image_list = []
    for path_input, path_output in tqdm(zip(input_list, target_list)):
        is_file(path_input, path_output)
        
        raster_rgb_numpy = np.asarray(Image.open(path_input)).transpose(2,0,1)
        raster_rgb_norm = ((raster_rgb_numpy.astype('float32') / 255.0) * 2.0) - 1.0

        target = load_labels_raster(path_output)
        if 'test' in phase:
            yield raster_rgb_norm, target, rasterio.open(path_output).meta
        else:
            yield raster_rgb_norm, target


def load_raster_multitask(input_list, target_list):
    import rasterio, os
    image_list = []
    for path_input, (path_output, path_output2) in tqdm(zip(input_list, target_list)):

        is_file(path_input, path_output, path_output2)
        
        raster_rgb_numpy = np.asarray(Image.open(path_input)).transpose(2,0,1)
        raster_rgb_norm = ((raster_rgb_numpy.astype('float32') / 255.0) * 2.0) - 1.0

        target = load_labels_raster(path_output, path_output2)
        yield raster_rgb_norm, target

def load_rgb_and_labels(input_list, target_path, phase, dfc_preprocessing, which_raster, use_semantics=False, save_semantics=False, normalize=False):
    """
    load all depths/labels to label_cache with crop
    Used for raster_regression and raster_multitask (regression+semantics)
    """
    import rasterio
    from rasterio.mask import mask
    from rasterio.plot import reshape_as_image
    import geopandas as gpd
    from shapely.geometry import box
    from scipy.misc import imresize

    # depth_image_raster = rasterio.open(target_path[0])
    # if len(target_path) == 2:
    #     dem_raster = rasterio.open(target_path[1])

    depth_raster = [rasterio.open(path) for path in target_path]
    if normalize:
        max_value, min_value = get_min_max(depth_raster, which_raster)

    print('Loading {} patches...'.format(phase))
    for i, img_input_path in enumerate(tqdm(input_list)):
        with rasterio.open(img_input_path) as raster_rgb:
            labels_patches = []
            # get bounds and crop from depth_image_raster
            bounds = raster_rgb.bounds
            bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0])
            coords = getFeatures(geo)
            labels_patches_masks = [mask(image_raster, shapes=coords, crop=True) for image_raster in depth_raster]

            depth_patch, depth_patch_transform = labels_patches_masks[0]
            
            # normalize data between -1 and 1 and append to cache
            raster_rgb_numpy = raster_rgb.read()
            raster_rgb_norm = ((raster_rgb_numpy.astype('float32') / 255.0) * 2.0) - 1.0
            pil_shape = (raster_rgb_norm.shape[-2:])[::-1]
            pil_depth_patch_shape = (depth_patch.shape[-2:])[::-1]
            
            if dfc_preprocessing == 1:
                pil_shape = [dim // 2 for dim in pil_shape] # Resize
                raster_rgb_norm = resize_rgb(raster_rgb_norm, pil_shape, Image.BILINEAR)
            elif dfc_preprocessing == 2: # make input same resolution as output
                pil_shape = [dim // 10 for dim in pil_shape] # Resize
                raster_rgb_norm = resize_rgb(raster_rgb_norm, pil_shape, Image.BILINEAR)

            # For test phase
            depth_patch_ = depth_patch

            # if len(labels_patches_masks) >= 2 and which_raster != 'dsm':
            if which_raster != 'dsm':
                depth_patch_dsm = depth_patch
                depth_patch_dem = labels_patches_masks[1][0]
                mask_dsm = (depth_patch_dsm < 9000) * (depth_patch_dem < 9000)
                not_mask_dsm = np.logical_not(mask_dsm)
                depth_patch = mask_dsm * (depth_patch_dsm - depth_patch_dem)
                depth_patch = depth_patch * ((not_mask_dsm * 10000) + 1)
            if normalize: # between [-1, 1]
                depth_patch = ((depth_patch - min_value) / (max_value - min_value)) * 2 - 1

            # if dfc_preprocessing == 0: # resize target
            depth_patch = resize(depth_patch[0], pil_shape, Image.BILINEAR) if dfc_preprocessing == 0 else depth_patch[0]
            labels_patches.append(depth_patch)
            if use_semantics:
                sem_label= labels_patches_masks[-1][0] 
                sem_label = resize(sem_label[0], pil_shape, Image.NEAREST) if dfc_preprocessing == 0 else sem_label[0]
                labels_patches.append(sem_label)

            if phase == 'test':
                out_meta = depth_raster[0].meta.copy()
                out_meta.update({"driver": "GTiff",
                "height": depth_patch_.shape[1],
                "width": depth_patch_.shape[2],
                "transform": depth_patch_transform}
                        )

                if save_semantics:
                    out_meta_sem = depth_raster[-1].meta.copy()
                    out_meta_sem.update({"driver": "GTiff",
                    "dtype": 'uint8',
                    "nodata": None,
                    "height": depth_patch_.shape[1],
                    "width": depth_patch_.shape[2],
                    "transform": depth_patch_transform}
                            )
                    out_meta = [out_meta, out_meta_sem]

                yield raster_rgb_norm, depth_patch, out_meta, pil_depth_patch_shape # To add labels for test
            else:
                yield raster_rgb_norm, labels_patches

def mask_invalid_depth(depth):
    # depth = depth # bigger than the minimum 
    # mask numbers > depth_max
    depth_mask = depth < 9000 # everest has 8.848m
    return depth * depth_mask # values to ignore will be 0 # thats ugly
