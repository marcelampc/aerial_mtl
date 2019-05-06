# based on https://github.com/pytorch/examples/blob/master/super_resolution/
# dataset.py
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

import torch.utils.data as data

import os
from os import listdir
from os.path import join
from PIL import Image
from ipdb import set_trace as st
import random
from math import pow
import numpy as np
from random import randint

from .dataset import make_dataset, load_img, str2bool
from .dataset import DatasetFromFolder as GenericDataset
from torch import cat
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop_tensor(tensor, params):
    i, j, h, w = params
    cropped_tensor = tensor[:, i : (i + h), j : (j + w)]
    return cropped_tensor

def get_params(tensor, crop_size=[224, 224]):
    w, h = tensor.size()[2], tensor.size()[1] # tensor
    tw, th = crop_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw

class DatasetFromFolder(GenericDataset):

    def __getitem__(self, index):
        self.crop = False
        state = random.getstate()
        max_rotation = 5.0
        # load and apply transformation/ or not
        input_path, target_path = self.input_target_list[index]
            
        img_input = load_img(input_path)
        img_target = load_img(target_path)

        # if self.DA_hflip:
        prob_hflip = random.random()
        prob_rotation = np.random.normal(0, max_rotation / 3.0)
        prob_scale = np.random.uniform(1.0, 1.5)

        random.setstate(state)
        img_global_tensor = self.apply_image_transform(img_input, prob_hflip, prob_rotation, prob_scale)

        random.setstate(state)
        target_global_tensor = self.apply_image_transform(img_target, prob_hflip, prob_rotation, prob_scale)

        # Now we can crop to create the input to the local network (cropping tensor, not image...)
        crop_params = get_params(img_global_tensor, crop_size=[self.imageSize[0], self.imageSize[1]])

        img_local_tensor = crop_tensor(img_global_tensor, crop_params)
        target_local_tensor = crop_tensor(target_global_tensor, crop_params)

        return img_global_tensor, img_local_tensor, torch.Tensor(crop_params), target_global_tensor, target_local_tensor

# Have to prepare a main function to test each new dataset I create...