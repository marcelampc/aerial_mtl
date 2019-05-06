import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt

from .mtl_train import MultiTaskGen
from networks import networks

import numpy as np
from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import torch.nn.functional as F
from tqdm import tqdm

# regression
# Be able to add many loss functions
class RasterMTL(MultiTaskGen):
    def name(self):
        return 'Raster Multitask Model'

    def initialize(self, opt):
        MultiTaskGen.initialize(self, opt) # not necessary
   
    def apply_valid_pixels_mask(self, *data, value=0.0):
        # self.nomask_outG = data[0].data   # for displaying purposes
        mask = ((data[1].data > 0) & (data[1].data < 9000)).to(self.cuda, dtype=torch.float32)
        
        masked_data = []
        for d in data:
            masked_data.append(d * mask)

        return masked_data, mask.sum()
    
    def get_checkpoint(self, epoch):
        return ({'epoch': epoch,
                 'arch_netG': self.opt.net_architecture,
                 'state_dictG': self.netG.state_dict(),
                 'optimizerG': self.optimG,
                 'best_pred': self.best_val_error,
                 'dfc_preprocessing': self.opt.dfc_preprocessing,
                 'd_block_type': self.opt.d_block_type,
                 'which_raster': self.opt.which_raster,
                 'model': self.opt.model,
                 'tasks': self.opt.tasks,
                 'outputs_nc': self.opt.outputs_nc,
                 'mtl_method': self.opt.mtl_method,
                 'use_skips': self.opt.use_skips,
                 })

    def _train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        input_data = input_cpu.to(self.cuda)
        input_data.requires_grad = True
        self.set_current_visuals(input=input_data.data[0])
        batch_size = input_cpu.shape[0]

        outG = self.netG.forward(input_data)
        
        losses = []
        norm_grad = []
        for i_task, task in enumerate(self.opt.tasks):
            target = target_cpu[i_task].to(self.cuda)
            if task == 'semantics':
                losses.append(self.cross_entropy(outG[i_task], target))
                self.get_errors_semantics(target, outG[i_task], n_classes=self.opt.outputs_nc[i_task])
            elif task == 'depth':
                output = outG[i_task]
                # mask
                mask = ((target.data > 0) & (target.data < 9000)).to(self.cuda, dtype=torch.float32)
                target *= mask
                output *= mask
                losses.append(self.criterion_reg(target, output))
                self.get_errors_regression(target, output)

            # with torch.no_grad():
            #     gradients = torch.autograd.grad(losses[i_task], input_data, create_graph=True, retain_graph=True, allow_unused=False)[0].view(batch_size,-1) # check allow used
            #     norm_grad.append(self.to_numpy(gradients.norm(2,dim=1).mean())) # mean on batch
            #     del gradients

        self.loss_error = sum(losses)

        self.optimG.zero_grad()
        self.loss_error.backward()
        self.optimG.step()

        self.n_iterations += 1 # outG[0].shape[0]

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), loss_task)
            # self.norm_grad_sum += np.array(norm_grad).mean()
            # norm_grad = self.norm_grad_sum / self.n_iterations
            # self.set_current_errors(norm_grad=norm_grad)
