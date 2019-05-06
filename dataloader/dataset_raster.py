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

from .dataset_raster_utils import load_rgb_and_labels, load_rgb_and_label, get_paths_list, get_random_pos, sliding_window, normalized_rgb_to_pil_image

class DatasetFromFolder(data.Dataset):
    # def __init__(self, opt, root, phase, data_split=None, data_augmentation=["F", "F", "F", "F", "F"], crop=True, resize=True, data_transform=None, imageSize=[256], outputSize=0, dataset_name='nyu', ddff_dataset=None, cache=True):
    def __init__(self, opt, root, phase, data_split, data_augmentation, crop=True, resize=True, data_transform=None, imageSize=[256], outputSize=0, dataset_name='nyu'):
        super(DatasetFromFolder, self).__init__()
        
        # self.augmentation = augmentation
        self.imageSize = imageSize if len(imageSize) == 2 else imageSize * 2        

        # List of files
        self.input_list, self.target_path = get_paths_list(opt, data_split, phase, opt.dataset_name)
        
        self.data_cache_ = []
        self.labels_cache_ = {} # dictionary
        self.meta_data_= []
        self.depth_shapes_ = []
        self.phase = phase

        self.opt = opt
        self.tasks = self.opt.tasks

        # dfc_preprocessing:
        # . 0: patches of 320x320 input.shape = output.shape
        # . 1: patches of 512x512 and resize to 256x256
        self.dfc_preprocessing = opt.dfc_preprocessing
        self.data_augmentation = data_augmentation
        self.crop = False
        self.resize = False
        max_r = opt.max_rotation
        mean_r = opt.mean_rotation
        self.data_augm_obj = DataAugmentation(data_augmentation, self.crop, self.resize, self.imageSize, opt.scale_to_mm, mean_rotation=mean_r, max_rotation=max_r, data_transform=data_transform, datatype='height')

        self.load_tiles(phase)

    def load_tiles(self, phase):
        if self.opt.dataset_name == 'dfc':
            use_semantics = ('semantics' in self.tasks)
            # select right load_data
            if use_semantics & len(self.tasks) == 1:
                from .dataset_raster import load_rgb_and_label as load_data
                [self.append_data_to_cache(rgb, labels) for rgb, labels in load_data(self.input_list, target_path = self.target_path, phase=phase, dfc_preprocessing=self.dfc_preprocessing)]
            else:
                from .dataset_raster_utils import load_rgb_and_labels as load_data
                [self.append_data_to_cache(rgb, labels) for rgb, labels in load_data(self.input_list, self.target_path, phase, self.dfc_preprocessing, which_raster=self.opt.which_raster, use_semantics=use_semantics)]
        elif self.opt.dataset_name == 'vaihingen':
            if self.opt.model == 'multitask':
                from .dataset_raster_utils import load_raster_multitask as load_raster
            else:
                from .dataset_raster_utils import load_raster # height
            [self.append_data_to_cache(rgb, labels) for rgb, labels in load_raster(self.input_list, self.target_path)]

    def __len__(self):
        if self.phase == 'train':
            # Default epoch size is 10 000 samples
            return 10000
        else:
            return len(self.data_cache_)

    def append_data_to_cache(self, rgb, labels, meta_data=None):
        if 'val' in self.phase:
            # for each image, create slidding windows and append to cache
            [self.data_cache_.append(rgb[:, y1:y2, x1:x2]) for x1, x2, y1, y2 in sliding_window(rgb, self.imageSize, self.imageSize)]
            for i, task in enumerate(self.tasks):
                # self.labels_cache_[task] = [] if not self.labels_cache_[task]
                [self.labels_cache_[task].append(labels[i][y1:y2, x1:x2]) for x1, x2, y1, y2 in sliding_window(labels[i], self.imageSize, self.imageSize)]
        else:
            self.data_cache_.append(rgb) # in numpy. Do I have to save meta data?
            for i, task in enumerate(self.tasks):
                try:
                    self.labels_cache_[task].append(labels[i])
                except:
                    self.labels_cache_[task] = []
                    self.labels_cache_[task].append(labels[i])

        if meta_data is not None:
            self.meta_data_.append(meta_data) 

    def __getitem__(self, index): # index is not used
        # Get a random patch
        if self.phase == 'train':
            # Pick a random image
            random_idx = random.randint(0, len(self.input_list) - 1)
            # All data is already loaded
            data = self.data_cache_[random_idx]
            
            self.data_augm_obj.set_probabilities()
            
            x1, x2, y1, y2 = get_random_pos(data, self.imageSize)
            data = data[:,y1:y2,x1:x2]
            data = self.data_augm_obj.apply_image_transform(normalized_rgb_to_pil_image(data))[0]

            labels = [self.data_augm_obj.apply_image_transform(self.labels_cache_[task][random_idx][y1:y2,x1:x2])[0] for task in self.tasks]


        else: # validation use sliding window
            data = self.data_cache_[index]
            labels = []
            for task in self.tasks:
                if 'semantics' in task:
                    labels.append(torch.from_numpy(self.labels_cache_[task][random_idx]))
                else:
                    labels.append(torch.from_numpy(self.labels_cache_[task][random_idx]).unsqueeze(0))
                    
        # Return the torch.Tensor values
        return (data,
                labels)