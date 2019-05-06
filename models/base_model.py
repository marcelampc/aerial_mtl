# Based on cycleGAN  -  Complete
import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util.visualizer import Visualizer

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = ''
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.create_save_folders()

        self.start_epoch = 1
        self.best_val_error = 999.9

        self.criterion_eval = nn.L1Loss()

        self.input = self.get_variable(torch.FloatTensor(self.batchSize, 3, self.opt.imageSize, self.opt.imageSize))
        self.target = self.get_variable(torch.FloatTensor(self.batchSize, 1, self.opt.imageSize, self.opt.imageSize))
        # self.logfile = # ToDo

        # visualizer
        self.visualizer = Visualizer(opt)

        # Logfile
        self.logfile = open(os.path.join(self.checkpoints_path, 'logfile.txt'), 'a')
        if opt.validate:
            self.logfile_val = open(os.path.join(self.checkpoints_path, 'logfile_val.txt'), 'a')

        # Prepare a random seed that will be the same for everyone
        opt.manualSeed = random.randint(1, 10000)   # fix seed
        print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        if opt.cuda:
            torch.cuda.manual_seed(opt.manualSeed)

        # uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms.
        cudnn.benchmark = True
        cudnn.enabled =   True

        if not opt.train and not opt.test:
            raise Exception("You have to set --train or --test")

        if torch.cuda.is_available and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should run WITHOUT --cpu")
        if not torch.cuda.is_available and opt.cuda:
            raise Exception("No GPU found, run WITH --cpu")

    def set_input(self, input):
        self.input = input

    def get_optimizer(self):
        pass

    def get_checkpoint(self, epoch):
        pass

    def update_learning_rate():
        pass

    def train_batch(self):
        """Each method has a different implementation"""
        pass

    def get_current_errors(self):
        pass

    def get_current_errors_display(self):
        pass

    def get_current_visuals(self):
        pass

    # def forward(self):
    #     pass
    # used in test time, no backprop
    # def test(self):
    #     pass

    # def get_image_paths(self):
    #     pass

    # def optimize_parameters(self):
    #     pass

    # def get_current_visuals(self):
    #     return self.input

    # def get_current_errors(self):
    #     return {}

    # def save(self, label):
    #     pass

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            self.start_epoch = checkpoint['epoch']
            self.opt.which_model_netG = checkpoint['arch_netG']
            netG = self.get_network()
            netG.load_state_dict(checkpoint['state_dictG'])
            best_val_error = checkpoint['best_pred']
            if self.opt.resume:
                print('Load optimizer(s)')

            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

    def create_save_folders(self):
        if self.opt.train:
            os.system('mkdir -p {0}'.format(self.checkpoints_path))
        if self.opt.save_samples:
            subfolders = ['input', 'target', 'results', 'output']
            self.save_samples_path = os.path.join('results/train_results/', self.opt.name)
            for subfolder in subfolders:
                path = os.path.join(self.save_samples_path, subfolder)
                os.system('mkdir -p {0}'.format(path))
            if self.opt.test:
                self.save_samples_path = os.path.join('results/test_results/', self.opt.name)
                self.save_samples_path = os.path.join(self.save_samples_path, self.opt.epoch)
                for subfolder in subfolders:
                    path = os.path.join(self.save_samples_path, subfolder)
                    os.system('mkdir -p {0}'.format(path))