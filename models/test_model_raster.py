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

from .mtl_test import MTL_Test as GenericTestModel

from rasterio.crs import CRS
from rasterio.transform import Affine
OUT_META_SEM = [{'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 271460.0,
       0.0, -0.5, 3290290.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 271460.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 272056.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1193, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 272652.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 273248.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 273844.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 274440.0,
       0.0, -0.5, 3290290.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 274440.0,
       0.0, -0.5, 3290891.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 275036.0,
       0.0, -0.5, 3290290.0)}, {'driver': 'GTiff', 'dtype': 'uint8', 'nodata': None, 'width': 1192, 'height': 1202, 'count': 1, 'crs': CRS.from_dict(init='epsg:26915'), 'transform': Affine(0.5, 0.0, 275036.0,
       0.0, -0.5, 3290891.0)}]

from dataloader.data_loader import CreateDataLoader

class TestModel(GenericTestModel):
    def initialize(self, opt):
        # GenericTestModel.initialize(self, opt)
        self.opt = opt
        self.get_color_palette()
        self.opt.imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        self.gpu_ids = ''
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.create_save_folders()
        self.opt.use_semantics = (('multitask' in self.opt.model) or ('semantics' in self.opt.model))

        self.netG = self.load_network()
        # self.opt.dfc_preprocessing = 2
        # self.data_loader, _ = CreateDataLoader(opt, Dataset)

        # visualizer
        self.visualizer = Visualizer(self.opt)
        if 'semantics' in self.opt.tasks:
            from util.util import get_color_palette
            self.opt.color_palette = np.array(get_color_palette(self.opt.dataset_name))
            # self.opt.color_palette = list(self.opt.color_palette.reshape(-1))
            # st()

    # def initialize(self, opt):
    #     GenericTestModel.initialize(self, opt)
    #     self.get_color_palette()

    def name(self):
        return 'Raster Test Model'

    def get_color_palette(self):
        if self.opt.dataset_name == 'dfc':
            self.opt.color_palette = [  
                                        [0, 0, 0], [0, 205, 0], [127, 255, 0], [46, 139, 87], [0, 139, 0], [0, 70, 0], [160, 82, 45], [0, 255, 255], [255, 255, 255], [216, 191, 216], [255, 0, 0], [170, 160, 150], [128, 128, 128], [160, 0, 0], [80, 0, 0], [232, 161, 24], [255, 255, 0], [238, 154, 0], [255, 0, 255], [0, 0, 255], [176, 196, 222]
                                        ]
        elif self.opt.dataset_name == 'isprs':
            self.opt.color_palette = [ [0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],                                      ]

    def load_network(self):
        if self.opt.epoch is not 'latest' or self.opt.epoch is not 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = int(checkpoint['epoch'])
            self.opt.net_architecture = checkpoint['arch_netG']
            try:
                self.opt.d_block_type = checkpoint['d_block_type']
                # Extra options for raster:
                self.opt.which_raster = checkpoint['which_raster']
                self.opt.model = checkpoint['model']
                self.opt.tasks = checkpoint['tasks']
                self.opt.outputs_nc = checkpoint['outputs_nc']
                self.opt.n_classes = checkpoint['n_classes']
            except:
                pass
            self.opt.use_skips = checkpoint['use_skips']
            self.opt.model = checkpoint['model']
            self.opt.dfc_preprocessing = checkpoint['dfc_preprocessing']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            self.opt.outputs_nc = checkpoint['outputs_nc']
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

    def save_raster_png(self, data, filename):
        if 'semantics' in filename:
            from util.util import labels_to_colors
            image_save = Image.fromarray(np.squeeze(labels_to_colors(data, self.opt.color_palette).astype(np.int8)), mode='RGB').convert('P', palette=Image.ADAPTIVE, colors=256)
            image_save.save(filename)

    def save_merged_rasters(self, datatype, fileroot=None):
        import rasterio
        from rasterio.merge import merge
        from rasterio.plot import show
        from os.path import join
        import argparse
        import glob
        
        if fileroot == None:
            fileroot = datatype

        root = '{}/{}/{}*.tif'.format(self.save_samples_path, datatype, fileroot)
        filename = '{}/{}/merged_{}.tif'.format(self.save_samples_path, datatype, fileroot)

        files = glob.glob(join(root))
        mosaic_rasters = [rasterio.open(file) for file in files]

        mosaic, out_transform = merge(mosaic_rasters)

        meta = (rasterio.open(files[0])).meta

        meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width":  mosaic.shape[2],
                    "transform": out_transform})

        with rasterio.open(filename, "w", **meta) as dest:
            dest.write(mosaic)

        filename = '{}/{}/{}_merged.png'.format(self.save_samples_path, datatype, datatype) 
        self.save_raster_png(mosaic, filename)

        if 'output' in filename or 'target' in filename:
            self.save_height_colormap(filename, mosaic)

    def test_raster(self):
        if 'semantics' in self.opt.model:
            from dataloader.dataset_raster import load_rgb_and_label as load_data
            self.test_raster_notarget(load_data)
        else:
            from dataloader.dataset_raster import load_rgb_and_labels as load_data
            self.test_raster_target(load_data)

    def initialize_test_bayesian(self, opt):
        from dataloader.dataset_bank import dataset_dfc
        print('Test phase using {} split.'.format(self.opt.test_split))
        phase = 'test'

        input_list, target_path = dataset_dfc(self.opt.dataroot, data_split=self.opt.test_split, phase='test', model=self.opt.model, which_raster=self.opt.which_raster)

        # Sanity check : raise an error if some files do not exist
        for f in input_list + target_path:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        from dataloader.dataset_raster import load_rgb_and_labels as load_data
        self.data_loader = [(rgb, depth, meta, depth_patch_shape) for rgb, depth, meta, depth_patch_shape in load_data(input_list, target_path, phase, self.opt.dfc_preprocessing, which_raster=self.opt.which_raster, use_semantics=False, save_semantics=self.opt.save_semantics)] # false because we do not have the GT        
        # no error in save semantics, same value to both variables
        self.netG.eval()
        self.netG.apply(self.activate_dropout)

    def activate_dropout(self, m):
        if type(m) == nn.Dropout:
            # print(m)
            m.train()

    def get_meta_data(self):
        return self.meta_data

    def get_shape(self):
        return self.shape

    def get_data_loader_size(self):
        return len(self.data_loader)

    def test_bayesian(self, it, n_iters):
        error_list = []
        outG_list = []

        use_semantics = self.opt.use_semantics
        self.opt.use_semantics = False
        # self.augmentation = augmentation
        imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        test_stride = self.opt.test_stride if len(self.opt.test_stride) == 2 else self.opt.test_stride * 2

        # create a matrix with a gaussian distribution to be the weights during reconstruction
        prob_matrix = self.gaussian_kernel(imageSize[0], imageSize[1])

        # for it, (input, target, meta_data, depth_patch_shape) in enumerate(tqdm(self.data_loader)):
        input, target, meta_data, depth_patch_shape = self.data_loader[it]
        for it in (tqdm(range(n_iters))):
            rgb_cache = []
            depth_cache = []
            self.meta_data = meta_data
            self.shape = depth_patch_shape
            # pred = np.zeros(input.shape[-2:])
            # concatenate probability matrix
            pred = np.zeros([input.shape[-2], input.shape[-1]])
            if self.opt.reconstruction_method == 'gaussian':
                pred = np.zeros([2, input.shape[-2], input.shape[-1]])
                pred_sem = np.zeros([self.opt.n_classes, input.shape[-2], input.shape[-1]])
            else:
                pred_sem = np.zeros([input.shape[-2], input.shape[-1]])
            target_reconstructed = np.zeros(input.shape[-2:])

            # input is a tensor
            rgb_cache = [crop for crop in self.sliding_window_coords(input, test_stride, imageSize)]
            depth_cache = [crop for crop in self.sliding_window_coords(target, test_stride, imageSize)] # don't need both

            for input_crop_tuple, target_crop_tuple in tqdm(zip(rgb_cache, depth_cache), total=len(rgb_cache)):
                input_crop, (x1, x2, y1, y2) = input_crop_tuple
                input_crop = self.get_variable(input_crop)
                # self.complete_padding = True
                # ToDo: Deal with padding later
                if self.opt.use_padding:
                    from torch.nn import ReflectionPad2d

                    self.opt.padding = self.get_padding_image_dims(input_crop)

                    input_crop = ReflectionPad2d(self.opt.padding)(input_crop)
                    (pwl, pwr, phu, phb) = self.opt.padding
                    # target_crop = ReflectionPad2d(self.opt.padding)(target_crop)

                with torch.no_grad():
                    outG, _ = self.netG.forward(input_crop)

                out_numpy = outG.data[0].cpu().float().numpy()
                if self.opt.reconstruction_method == 'concatenation':
                    if self.opt.use_padding:
                        pred[y1:y2,x1:x2] = (out_numpy[0])[phu:phu+self.opt.imageSize[1], pwl:pwl+self.opt.imageSize[0]]
                    else:
                        pred[y1:y2,x1:x2] = out_numpy[0]
                elif self.opt.reconstruction_method == 'gaussian':
                    pred[0,y1:y2,x1:x2] += np.multiply(out_numpy[0], prob_matrix)
                    pred[1,y1:y2,x1:x2] += prob_matrix
                
                target_reconstructed[y1:y2,x1:x2] = target_crop_tuple[0]

            if self.opt.reconstruction_method == 'gaussian':
                gaussian = pred[1]
                pred = np.divide(pred[0], gaussian)
                # pred_sem = np.divide(pred_sem, gaussian)
                
                # st()
                if self.opt.dfc_preprocessing == 0:
                    # resize outputs
                    pred = np.array(Image.fromarray(pred).resize((pred.shape[1]//10, pred.shape[0]//10), Image.BILINEAR))
                    target_reconstructed = np.array(Image.fromarray(target_reconstructed).resize((target_reconstructed.shape[1]//10, target_reconstructed.shape[0]//10), Image.BILINEAR))
                error_list.append(np.abs(pred - target_reconstructed))
                outG_list.append(np.abs(pred))
            
        return error_list, outG_list, target_reconstructed

    def test_raster_notarget(self, load_data):
        from dataloader.dataset_bank import dataset_dfc
        print('Test phase using {} split.'.format(self.opt.test_split))
        phase = 'test'

        imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        test_stride = self.opt.test_stride if len(self.opt.test_stride) == 2 else self.opt.test_stride * 2
        input_list = dataset_dfc(self.opt.dataroot, data_split=self.opt.test_split, phase='test', model=self.opt.model)

        # Sanity check : raise an error if some files do not exist
        for f in input_list:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        data_loader = [(rgb) for rgb in load_data(input_list, phase, dfc_preprocessing=self.opt.dfc_preprocessing)] 
        
        self.netG.eval()

        # create a matrix with a gaussian distribution to be the weights during reconstruction
        prob_matrix = self.gaussian_kernel(imageSize[0], imageSize[1])
        
        for it, input in enumerate(tqdm(data_loader)):
            rgb_cache = []

            pred_gaussian = np.zeros([input.shape[-2], input.shape[-1]])
            if self.opt.reconstruction_method == 'gaussian':
                pred_sem = np.zeros([self.opt.n_classes, input.shape[-2], input.shape[-1]])
            else:
                pred_sem = np.zeros([input.shape[-2], input.shape[-1]])
            target_reconstructed = np.zeros(input.shape[-2:])

            # input is a tensor
            rgb_cache = [crop for crop in self.sliding_window_coords(input, test_stride, imageSize)]

            for input_crop_tuple in tqdm(rgb_cache, total=len(rgb_cache)):
                input_crop, (x1, x2, y1, y2) = input_crop_tuple
                input_crop = self.get_variable(input_crop)
                
                # ToDo: Deal with padding later
                if self.opt.use_padding:
                    from torch.nn import ReflectionPad2d

                    self.opt.padding = self.get_padding_image_dims(input_crop)

                    input_crop = ReflectionPad2d(self.opt.padding)(input_crop)
                    (pwl, pwr, phu, phb) = self.opt.padding
                    # target_crop = ReflectionPad2d(self.opt.padding)(target_crop)

                with torch.no_grad():
                    outG_sem = self.netG.forward(input_crop)
                
                if self.opt.reconstruction_method == 'gaussian':
                    outG_sem_prob = nn.Sigmoid()(outG_sem)
                    seg_map = outG_sem_prob.cpu().data[0].numpy()
                    pred_sem[:, y1:y2,x1:x2] += np.multiply(seg_map, prob_matrix)
                    pred_gaussian[y1:y2,x1:x2] += prob_matrix
                else:
                    pred_sem[y1:y2,x1:x2] = np.argmax(outG_sem.cpu().data[0].numpy(), axis=0)

                # visualize 
                visuals = OrderedDict([('input', input_crop.data),
                            ('out_sem', np.argmax(outG_sem.cpu().data[0].numpy(), axis=0))
                            ])
                self.display_test_results(visuals)

            if self.opt.save_samples:
                if self.opt.reconstruction_method == 'gaussian':
                    pred_sem = np.divide(pred_sem, pred_gaussian)
                self.save_raster_images_semantics_only(input, pred_sem, index=it+1, phase='test')

    def test_raster_target(self, load_data):
        from dataloader.dataset_bank import dataset_dfc
        print('Test phase using {} split.'.format(self.opt.test_split))
        phase = 'test'

        use_semantics = self.opt.use_semantics
        self.opt.use_semantics = False
        # self.augmentation = augmentation
        imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        test_stride = self.opt.test_stride if len(self.opt.test_stride) == 2 else self.opt.test_stride * 2
        input_list, target_path = dataset_dfc(self.opt.dataroot, data_split=self.opt.test_split, phase='test', model=self.opt.model, which_raster=self.opt.which_raster)

        # Sanity check : raise an error if some files do not exist
        for f in input_list + target_path:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        data_loader = [(rgb, depth, meta, depth_patch_shape) for rgb, depth, meta, depth_patch_shape in load_data(input_list, target_path, phase, self.opt.dfc_preprocessing, which_raster=self.opt.which_raster, use_semantics=False, save_semantics=self.opt.save_semantics)] # false because we do not have the GT        
        # no error in save semantics, same value to both variables
        self.netG.eval()

        # if self.opt.normalize:
        #     from dataloader.dataset_raster import get_min_max
        #     import rasterio
        #     max_v, min_v = get_min_max([rasterio.open(path) for path in target_path], self.opt.which_raster)

        # create a matrix with a gaussian distribution to be the weights during reconstruction
        prob_matrix = self.gaussian_kernel(imageSize[0], imageSize[1])
        # st()
        time_array = np.zeros(len(data_loader))

        for it, (input, target, meta_data, depth_patch_shape) in enumerate(tqdm(data_loader)):
            rgb_cache = []
            depth_cache = []
            start = time.time()
            
            # pred = np.zeros(input.shape[-2:])
            # concatenate probability matrix
            pred = np.zeros([input.shape[-2], input.shape[-1]])
            if self.opt.reconstruction_method == 'gaussian':
                pred = np.zeros([2, input.shape[-2], input.shape[-1]])
                pred_sem = np.zeros([self.opt.n_classes, input.shape[-2], input.shape[-1]])
            else:
                pred_sem = np.zeros([input.shape[-2], input.shape[-1]])
            target_reconstructed = np.zeros(input.shape[-2:])

            # input is a tensor
            rgb_cache = [crop for crop in self.sliding_window_coords(input, test_stride, imageSize)]
            depth_cache = [crop for crop in self.sliding_window_coords(target, test_stride, imageSize)] # don't need both
            # import cProfile
            # torch.cuda.synchronize()
            for input_crop_tuple, target_crop_tuple in tqdm(zip(rgb_cache, depth_cache), total=len(rgb_cache)):
                # cp = cProfile.Profile()
                # cp.enable()
            # for input_crop_tuple, target_crop_tuple in zip(rgb_cache, depth_cache):
                input_crop, (x1, x2, y1, y2) = input_crop_tuple
                input_crop = self.get_variable(input_crop)

                with torch.no_grad():
                    if 'multitask' in self.opt.model:
                        outG, outG_sem = self.netG.forward(input_crop)
                    else:
                        outG = self.netG.forward(input_crop)[0]
                out_numpy = outG.data[0].cpu().float().numpy()
                if self.opt.reconstruction_method == 'concatenation':
                    pred[y1:y2,x1:x2] = out_numpy[0]
                elif self.opt.reconstruction_method == 'gaussian':
                    pred[0,y1:y2,x1:x2] += np.multiply(out_numpy[0], prob_matrix)
                    pred[1,y1:y2,x1:x2] += prob_matrix
                
                if self.opt.save_semantics:
                    # pred_sem[:,y1:y2,x1:x2] += outG_sem.cpu().data[0].numpy()
                    if self.opt.reconstruction_method == 'gaussian':
                        # seg_map = np.argmax(outG_sem.cpu().data[0].numpy(), axis=0)
                        # pred_sem[y1:y2,x1:x2] += np.multiply(seg_map, prob_matrix)
                        outG_sem_prob = nn.Sigmoid()(outG_sem)
                        seg_map = outG_sem_prob.cpu().data[0].numpy()
                        pred_sem[:, y1:y2,x1:x2] += np.multiply(seg_map, prob_matrix)
                    else:
                        pred_sem[y1:y2,x1:x2] = np.argmax(outG_sem.cpu().data[0].numpy(), axis=0)

                    # visualize takes a lot of time
                    # visuals = OrderedDict([('input', input_crop.data),
                    #             # ('gt', target_crop.data),
                    #             ('output', outG), 
                    #             # ('gt_sem', self.target_sem.data[0].cpu().float().numpy()),
                    #             ('out_sem', np.argmax(outG_sem.cpu().data[0].numpy(), axis=0))
                    #             ])
                    # self.display_test_results(visuals)

                target_reconstructed[y1:y2,x1:x2] = target_crop_tuple[0]
                # break
                # st()
                # cp.disable()
                # cp.print_stats()
                # st()
            end = time.time()
            time_array[it] = end-start
            print('Time in seconds: {}'.format(end-start))
            t_time = end-start
            day = t_time // (24 * 3600)
            t_time = t_time % (24 * 3600)
            hour = t_time // 3600
            t_time %= 3600
            minutes = t_time // 60
            t_time %= 60
            seconds = t_time
            print("d:h:m:s-> %d:%d:%d:%d" % (day, hour, minutes, seconds))

            if self.opt.save_samples:
                if self.opt.reconstruction_method == 'gaussian':
                    gaussian = pred[1]
                    pred = np.divide(pred[0], gaussian)
                    pred_sem = np.divide(pred_sem, gaussian)
                    # if self.opt.normalize:
                    #     target_reconstructed = (target_reconstructed + 1) / 2
                    #     target_reconstructed = (target_reconstructed * (max_v - min_v)) + min_v
                    #     pred = (pred + 1) / 2
                    #     pred = (pred * (max_v - min_v)) + min_v
                    # print('Target: max[{}] min[{}]'.format(target_reconstructed.max(),target_reconstructed.min()))
                    # print('Pred: max[{}] min[{}]'.format(pred.max(),pred.min()))

                indexplus = 4
                if self.opt.save_semantics:
                    self.save_raster_images_semantics(input, pred, target_reconstructed, pred_sem, meta_data, depth_patch_shape, it + 1 + indexplus, 'test')
                else:
                    self.save_raster_images(input, pred, target_reconstructed, meta_data, depth_patch_shape, it + 1 + indexplus, 'test')
                del input, pred, target_reconstructed, gaussian, pred_sem

        print('Test statistics')
        print('Mean and standard deviation in seconds {:.3f} {:.3f}'.format(np.mean(time_array), np.std(time_array)))

        print('Saving merged!')
        self.save_merged_rasters('output')
        self.save_merged_rasters('target')
        self.save_merged_rasters('semantics')

    def display_test_results(self, visuals):
        self.visualizer.display_images(visuals, 1)

    def get_padding(self, dim):
        final_dim = (dim // 32 + 1) * 32
        return final_dim - dim

    def sliding_window_coords(self, data, step, window_size):
        from dataloader.dataset_raster import sliding_window
        # data = data.data[0].cpu().float().numpy()
        for x1, x2, y1, y2 in sliding_window(data, step, window_size):
            if len(data.shape) == 2:
                yield (data[y1:y2,x1:x2], [x1,x2,y1,y2])
            else:         
                yield (torch.from_numpy(data[:, y1:y2,x1:x2]).unsqueeze(0), [x1,x2,y1,y2]) # why do I have to unsqueeze here?

    def gaussian_kernel(self, width, height, sigma=0.2, mu=0.0):
        x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, height))
        d = np.sqrt(x*x+y*y)
        gaussian_k = (np.exp(-((d-mu)**2 / (2.0 * sigma**2)))) / np.sqrt(2 * np.pi * sigma**2)
        return gaussian_k # / gaussian_k.sum()

    def get_padding_image_dims(self, img):
        # get tensor dimensions
        h, w = img.size()[2:]
        # self.opt.imageSize = (w + 4, h + 4)
        w_pad, h_pad = self.get_padding(w+4)+4, self.get_padding(h+4)+4

        pwr = w_pad // 2
        pwl = w_pad - pwr
        phb = h_pad // 2
        phu = h_pad - phb

        # pwl, pwr, phu, phb
        return (pwl, pwr, phu, phb)

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

    def save_height_colormap(self, filename, data, cmap='jet'):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        dpi = 80
        data = data[0,:,:]
        height, width = data.shape 
        figsize = width / float(dpi), height / float(dpi)
        # change string
        if 'output' in filename:
            filename = filename.replace('merged_output', 'cmap_merged_output')
        else:
            filename = filename.replace('merged_target', 'cmap_merged_target')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        cax = ax.imshow(data, vmax=30, vmin=0, aspect='auto', interpolation='spline16', cmap=cmap)
        ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

        fig.savefig(filename, dpi=dpi)
        del fig, data

    def save_dsm_as_raster(self, data, filename, meta_data, shape):
        import rasterio
        filename = os.path.join(self.save_samples_path, filename)
        depth_patch = np.expand_dims(np.array(Image.fromarray(data).resize(shape, Image.BILINEAR)), axis=0)

        # if 'output' in filename or 'target' in filename:
        #     try:
        #         self.save_height_colormap(filename, depth_patch, cmap='jet')
        #     except:
        #         pass
        with rasterio.open(filename, "w", **meta_data) as dest:
            if dest.write(depth_patch) == False:
                print('Couldnt save image, sorry')

    def save_raster_images_semantics(self, input, output, target, semantics, meta_data, shape, index, phase='train', out_type='png'):
        from dataloader.dataset_raster import sliding_window
        self.save_raster_images(input, output, target, meta_data[0], shape, index, phase)
        del input, output, target
        import gc
        gc.collect()
        filename = '{}/semantics/semantics_{:04}.tif'.format(self.save_samples_path, index)

        if self.opt.reconstruction_method == 'gaussian':
            semantics = np.argmax(semantics, axis=0)

        semantics = np.array(semantics, dtype=np.uint8)
        sem_patch = np.expand_dims(np.array(Image.fromarray(semantics, mode='P').resize(shape, Image.NEAREST)), axis=0)
        del semantics

        import rasterio
        with rasterio.open(filename, "w", **meta_data[1]) as dest:
            if dest.write(sem_patch) == False:
                print('Couldnt save image, sorry')
            # base_stride /= 2
        del sem_patch

    # def save_height_colormap(self, filename, data, cmap='jet'):
    #     import matplotlib.pyplot as plt
    #     plt.switch_backend('agg')
    #     dpi = 80
    #     data = data[0,:,:]
    #     height, width = data.shape 
    #     figsize = width / float(dpi), height / float(dpi)
    #     # change string
    #     filename = filename.replace('output_', 'cmap_output_')
    #     fig = plt.figure(figsize=figsize)
    #     ax = fig.add_axes([0, 0, 1, 1])
    #     ax.axis('off')
    #     cax = ax.imshow(data, vmax=30, vmin=0, aspect='auto', interpolation='spline16', cmap=cmap)
    #     ax.set(xlim=[0, width], ylim=[height, 0], aspect=1)

    #     fig.savefig(filename, dpi=dpi)

    def save_raster_images_semantics_only(self, input, semantics, index, shape=(1192,1202), phase='train', out_type='png'):
        print('Saving semantics...')
        import gc
        gc.collect()
        
        filename = '{}/semantics/semantics_{:04}.tif'.format(self.save_samples_path, index)
        # 1192
        # 1202
        import time
        start = time.time()
        if self.opt.reconstruction_method == 'gaussian':
            semantics = np.argmax(semantics, axis=0)
        end = time.time()
        print('Time to argmax: {}'.format(end-start))

        semantics = np.array(semantics, dtype=np.uint8)
        sem_patch = np.expand_dims(np.array(Image.fromarray(semantics, mode='P').resize(shape, Image.NEAREST)), axis=0)

        import rasterio
        with rasterio.open(filename, "w", **OUT_META_SEM[index-1]) as dest:
            if dest.write(sem_patch) == False:
                print('Couldnt save image, sorry')

    def save_raster_images(self, input, output, target, meta_data, shape, index, phase='train', out_type='png'):
        # self.save_rgb_raster(input.data, '{}/input/input_{:04}.png'.format(self.save_samples_path, index))
        self.save_dsm_as_raster(output, 'output/output_{:04}.tif'.format(index), meta_data, shape)
        self.save_dsm_as_raster(target, 'target/target_{:04}.tif'.format(index), meta_data, shape)
        # self.save_dsm_as_raster(target.data, '{}/target/target_{:04}.tif'.format(self.save_samples_path, index), meta_data)

    def get_variable(self, tensor):
        variable = Variable(tensor)
        if self.opt.cuda:
            return variable.cuda()

    def create_save_folders(self, subfolders=['input','target','results','output','semantics']):
        if self.opt.save_samples:
            if self.opt.test:
                self.save_samples_path = os.path.join('results/{}'.format(self.opt.dataset_name), self.opt.name, self.opt.epoch)
                for subfolder in subfolders:
                    path = os.path.join(self.save_samples_path, subfolder)
                    os.system('mkdir -p {0}'.format(path))
