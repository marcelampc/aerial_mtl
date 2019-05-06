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
from .dataset import get_paths_list, make_dataset, load_img, str2bool
from .dataset import DatasetFromFolder as GenDataset

from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_bin(path, dim):
    img_np = np.fromfile(path, dtype='>f')
    img_np = img_np.reshape(dim).transpose() * 1000
    img_np = img_np.astype(np.uint16).copy()

    # transform in PIL image
    img_pil = Image.fromarray(img_np, mode='I;16').convert('I')
    return img_pil

class DatasetFromFolder(GenDataset):

    def __getitem__(self, index):
        state = random.getstate()
        max_rotation = 5.0
        # load and apply transformation/ or not
        input_path, target_path, sem_path = self.input_target_list[index]
        img_input = load_img(input_path)
        img_target = load_bin(target_path, (640, 480))
        img_sem = load_img(sem_path)

        if self.crop:
            i, j, h, w = self.get_params(img_target, crop_size=[self.imageSize[0], self.imageSize[1]])
        else:
            i, j, h, w = 0, 0, 0, 0

        # if self.DA_hflip:
        prob_hflip = random.random()
        prob_rotation = np.random.normal(0, max_rotation / 3.0)
        prob_scale = np.random.uniform(1.0, 1.5)

        random.setstate(state)
        img_input_tensor = self.apply_image_transform(img_input, prob_hflip, prob_rotation, prob_scale, i, j, h, w)
        random.setstate(state)
        img_target_tensor = self.apply_image_transform(img_target, prob_hflip, prob_rotation, prob_scale, i, j, h, w)
        random.setstate(state)
        img_sem_tensor = self.apply_image_transform(img_sem, prob_hflip, prob_rotation, prob_scale, i, j, h, w)

        return img_input_tensor, img_target_tensor, img_sem_tensor
