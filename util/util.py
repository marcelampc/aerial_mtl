# Utilities
import torch
import shutil
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
from ipdb import set_trace as st
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
from PIL import Image
from torch import nn

save_samples_path = None

# from wGAN
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def print_weights(network):
    first = True
    for m in network.modules():
        if isinstance(m, nn.Conv2d):
            if first:
                first = False
                print('weight: {}'.format(m.weight.data))
                st()

def save_graphs(fig, name):
    fig.savefig(os.path.join(save_samples_path, name))

class bcolors:
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colors_to_labels(data, colors):
    """Convert colors to labels."""
    labels = np.zeros(data.shape[:2], dtype=int)

    for id_col, col in enumerate(colors):
        d = (data[:, :, 0] == col[0])
        d = np.logical_and(d, (data[:, :, 1] == col[1]))
        d = np.logical_and(d, (data[:, :, 2] == col[2]))
        labels[d] = id_col
    return labels.astype('uint8')

def labels_to_colors(labels, colors):
        """Convert labels to colors."""
        data = np.zeros(labels.shape+(3,))
        for id_col, col in enumerate(colors):
            d = (labels == id_col)
            data[d] = col
        return data

def get_color_palette(dataset_name):
    from .nyuv2.nyuv2_extra_data import colors as colors_nyu
    from .dfc.dfc_extra_data import colors as colors_dfc
    from .vaihingen.vaihingen_extra_data import colors as colors_vaihingen
    color_palette_dic={"vaihingen": colors_vaihingen,
                    "dfc": colors_dfc,
                    "nyu": colors_nyu,
                    }
    return color_palette_dic[dataset_name]

def get_dataset_semantic_weights(dataset_name):
    from .nyuv2.nyuv2_extra_data import weights as weights_nyu
    from .dfc.dfc_extra_data import weights as weights_dfc
    from .vaihingen.vaihingen_extra_data import weights as weights_vaihingen
    color_palette_dic={
                        "vaihingen": weights_vaihingen,
                        "dfc": weights_dfc,
                        "nyu": weights_nyu,
                        }
    return color_palette_dic[dataset_name]

colors_vaihingen = [ [0, 0, 0], [255, 255, 255], [0, 0, 255],
                     [0, 255, 255], [0, 255, 0], [255, 255, 0],
                     [255, 0, 0],
                   ]
