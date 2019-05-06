import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import numpy as np
from PIL import Image

import re

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util.visualizer import Visualizer
import networks.networks as networks

from .test_model import TestModel as GenericTestModel

class TestModel(GenericTestModel):
    def name(self):
        return 'Test Model'

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = checkpoint['epoch']
            self.opt.which_model_netG = checkpoint['arch_netG']
            netG = self.create_G_network()
            pretrained_dict = checkpoint['state_dictG']
            pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(pretrained_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    pretrained_dict[new_key] = pretrained_dict[key]
                    del pretrained_dict[key]
            netG.load_state_dict(pretrained_dict)
            if self.opt.cuda:
                netG = netG.cuda()
            self.best_val_error = checkpoint['best_pred']

            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG
        else:
            print("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

    def test(self, data_loader):
        print('Test phase using {} split.'.format(self.opt.test_split))
        epoch = 'test'
        data_iter = iter(data_loader)

        self.netG.eval()
        total_iter = 0

        for it, (input, target, target_2) in enumerate(tqdm(data_loader)):
            total_iter += 1
            
            input, target, target_2 = self.get_variable(input), self.get_variable(target), self.get_variable(target_2)

            # self.complete_padding = True
            if self.opt.use_padding:
                from torch.nn import ReflectionPad2d

                self.opt.padding = self.get_padding_image(input)

                input = ReflectionPad2d(self.opt.padding)(input)
                target = ReflectionPad2d(self.opt.padding)(target)
                target_2 = ReflectionPad2d(self.opt.padding)(target_2)

            with torch.no_grad():
                outG_1, outG_2 = self.netG.forward(input)

            if self.opt.save_samples:
                # visuals = OrderedDict([('input', input.data),
                #                       ('gt', target.data),
                #                       ('output', outG.data)])
                # visualizer.display_images(visuals, epoch)
                self.save_images(input, outG_1, outG_2, target, it + 1, 'test', out_type=self.get_type(self.opt.save_bin))

    def sliding_window_tensor(self, data, step, window_size):
        from dataloader.dataset_raster import sliding_window
        data_numpy = data.data[0].cpu().float().numpy()
        for x1, x2, y1, y2 in sliding_window(data_numpy, step, window_size):
            if len(data_numpy.shape) == 2:
                yield (data_numpy[x1:x2,y1:y2], [x1,x2,y1,y2])
            else:         
                yield (torch.from_numpy(data_numpy[:, x1:x2,y1:y2]).unsqueeze(0), [x1,x2,y1,y2]) # why do I have to unsqueeze here?

    def get_type(self, action):
        return {True: 'bin', False: 'png'}.get(action)

    def tensor2numpy(self, tensor, imtype=np.uint16):
        image_numpy = tensor.data[0].cpu().float().numpy()
        image_numpy = (image_numpy + 1) / 2.0
        # if self.opt.use_padding:
        #     image_width, image_height = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        #     d_left, _, d_upper, _ = self.opt.padding
        #     image_numpy = image_numpy[:, d_upper:(image_height + d_upper), d_left:(image_width + d_left)]
        #     # [-1,1] -- [0,255]
        #     image_numpy = np.transpose(image_numpy, (1, 2, 0))
        # if imtype == np.uint16:
        #     from math import pow
        #     mult = pow(2, 16) - 1
        # else:
        #     mult = 255
        #     mult = 1
            # image_numpy = image_numpy.astype(imtype)

        return image_numpy

    def get_padding(self, dim):
        final_dim = (dim // 32 + 1) * 32
        return final_dim - dim

    def get_padding_image(self, img):
        # get tensor dimensions
        h, w = img.size()[2:]
        self.opt.imageSize = (w, h)
        w_pad, h_pad = self.get_padding(w), self.get_padding(h)

        pwr = w_pad // 2
        pwl = w_pad - pwr
        phb = h_pad // 2
        phu = h_pad - phb

        # pwl, pwr, phu, phb
        return (pwl, pwr, phu, phb)

    def create_G_network(self):
        netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, 64, which_model_netG=self.opt.which_model_netG, opt=self.opt, norm='batch', gpu_ids='')
        # print(netG)
        return netG

    def create_save_folders(self):
        if self.opt.save_samples:
            subfolders = ['input', 'target', 'results', 'depth', 'sharp']
            # self.save_samples_path = os.path.join('results/eccv', self.opt.name)
            # for subfolder in subfolders:
            #     path = os.path.join(self.save_samples_path, subfolder)
            #     os.system('mkdir -p {0}'.format(path))
            if self.opt.test:
                self.save_samples_path = os.path.join('results/deblur', self.opt.name, self.opt.epoch)
                for subfolder in subfolders:
                    path = os.path.join(self.save_samples_path, subfolder)
                    os.system('mkdir -p {0}'.format(path))

    # def tensor2raster(self, tensor):


    def tensor2image(self, tensor, imtype=np.uint8, mode='RGB'):
        image_numpy = tensor[0].cpu().float().numpy()
        # [-1,1] -- [0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        mult = self.opt.scale_to_mm
        if self.opt.scale_to_mm == 0.0:
            if imtype == np.uint16:
                from math import pow
                mult = pow(2, 16) - 1
            else:
                mult = 255
        image_numpy = (image_numpy + 1) / 2.0 * mult
        image_numpy = image_numpy.astype(imtype)
        if image_numpy.shape[2] == 1:
            # st()
            image_numpy = np.repeat(image_numpy, 3, axis=2)
        return image_numpy

    def save_output_as_png(self, tensor, filename, imtype=np.uint8, mode='RGB', mode_2=''):
        # st()
        if self.opt.save_upsample:
            upsample_op = nn.Upsample(size=self.opt.upsample_size, mode='bilinear', align_corners=True)
            tensor = upsample_op(tensor)
        image_numpy = tensor[0].cpu().float().numpy()
        # [-1,1] -- [0,255]
        if imtype == np.uint16:
            from math import pow
            mult = pow(2, 16) - 1
        else:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            mult = 255
        if mode == 'RGB':
            # st()
            # image_numpy = np.transpose(image_numpy, (1, 2, 0))
            # image_numpy = np.transpose(image_numpy, (2, 1, 0))
            image_numpy = (image_numpy + 1) / 2.0 * mult
        else:
            image_numpy = image_numpy * self.opt.scale_to_mm # in milimeters
        image_numpy = image_numpy.astype(imtype)

        if mode_2 == 'I':
            image_save = Image.fromarray(np.squeeze(image_numpy), mode=mode).convert(mode=mode_2)
        else:
            image_save = Image.fromarray(np.squeeze(image_numpy), mode=mode)
            # st()

        if self.opt.use_padding:
            image_width, image_height = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
            d_left, _, d_upper, _ = self.opt.padding
            # left, upper, right, lower
            image_save = image_save.crop((d_left, d_upper, image_width + d_left, image_height + d_upper))
        image_save.save(filename)

    def save_images(self, input, output, output2, target, index, phase='train', out_type='png'):
        if out_type == 'png':
            # save other images
            self.save_output_as_png(input.data, '{}/input/input_{:04}.png'.format(self.save_samples_path, index))
            self.save_output_as_png(output.data, '{}/depth/depth_{:04}.png'.format(self.save_samples_path, index), imtype=np.uint16, mode='I;16', mode_2='I')
            self.save_output_as_png(output2.data, '{}/sharp/sharp_{:04}.png'.format(self.save_samples_path, index))
            self.save_output_as_png(target.data, '{}/target/target_{:04}.png'.format(self.save_samples_path, index), imtype=np.uint16, mode='I;16', mode_2='I')
        # concatenate input+target+output and save
        # if input.data.shape[1] > 3:   # case of focalstack
        #     input.data = input.data[:, :3
        # input = self.tensor2image(input.data)
        # target = self.tensor2image(target.data, imtype=np.uint8, mode='RGB')
        # output = self.tensor2image(output.data, imtype=np.uint8, mode='RGB')
        #
        # # st()
        # image_concat = np.concatenate((input, (np.concatenate((target, output), axis=1))), axis=1)
        # image_concat = Image.fromarray(np.squeeze(image_concat), mode='RGB')
        # st()
        # image_concat.save('{}/results/result_{:04}.png'.format(self.save_samples_path, index))


    def get_variable(self, tensor):
        variable = Variable(tensor)
        if self.opt.cuda:
            return variable.cuda()
