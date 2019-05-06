import os
import time
import torch
import torch.nn as nn
from torch.utils import data
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt
from tqdm import tqdm
from networks import networks
from util.visualizer import Visualizer
from dataloader.toy_dataset import RegressionDatasetOriginal as ToyDataset
from dataloader.toy_dataset import RegressionDatasetOnce as ToyDatasetVal
from dataloader.toy_dataset import RegressionDataset as ToyDatasetTrainTest # not the same batch

from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import numpy as np

def get_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        st()
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    return ave_grads
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)

# Be able to add many loss functions
class MTL_TOY():
    def name(self):
        return 'MultiTask for Toy Dataset'

    def initialize(self, opt):
        print(self.name())
        # set the random seeds for reproducibility
        self.random_seed = 123
        np.random.seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.manual_seed(self.random_seed)

        # MultiTaskGen.initialize(self, opt)
        self.opt = opt
        self.batchSize = opt.batchSize
        self.n_tasks = self.opt.n_tasks

        self.errors = OrderedDict()
        if self.opt.display_id > 0:
            self.visualizer = Visualizer(opt)
    
        # define the sigmas, the number of tasks and the epsilons
        # for the toy example
        if self.n_tasks == 2:
            sigmas = [1.0, float(opt.sigma)]
        elif self.n_tasks > 2:
            # sample from normal distribution
            pass
        print('Training toy example with sigmas={}'.format(sigmas))
        # n_tasks = len(sigmas)
        # B and epsilons are constant matrices
        # Information is shared in B
        # Epsilon contains task-specific information
        epsilons = np.random.normal(scale=3.5, size=(self.n_tasks, 100, 250)).astype(np.float32)
        B = np.random.normal(scale=10, size=(100, 250)).astype(np.float32)

        # initialize the data loader
        dataset = ToyDatasetTrainTest(sigmas, epsilons, B)
        # dataset = ToyDataset(sigmas, epsilons, B)
        # dataset_val = ToyDatasetVal(sigmas, epsilons, B)
        # self.opt.batchSize = 100
        
        self.data_loader = data.DataLoader(dataset, batch_size=self.opt.batchSize, num_workers=4, shuffle=False)
        # self.val_loader = data.DataLoader(dataset_val, batch_size=self.opt.batchSize, num_workers=4, shuffle=False)

        # Alpha here is the GP coefficient lambda in the paper
        self.alpha = 10.0 # self.opt.alpha
        self.netG = self.create_network()
        # lr=2e-4
        # if self.opt.optim == 'adam':
        #     self.optim = torch.optim.Adam(self.netG.parameters(), lr=lr)
        # elif 
        self.optim = self.get_optim()
        self.regression_loss = nn.L1Loss()
        self.cuda = torch.device('cuda:0')
        # self.first_epoch_test = True

    def get_optim(self, lr=2e-2):
        if self.opt.optim == 'adam':
            return torch.optim.Adam(self.netG.parameters(), lr=lr)
        elif self.opt.optim == 'sgd':
            return torch.optim.SGD(self.netG.parameters(), lr=lr)
        elif self.opt.optim == 'adagrad':
            return torch.optim.Adagrad(self.netG.parameters(), lr=lr)


    def create_network(self):
        from networks.mtl_toynetwork import ToyNetwork
        netG = ToyNetwork(self.opt.n_tasks, self.opt.mtl_method)
        if self.opt.cuda:
            netG = netG.cuda()
        return netG

    def restart_variables(self):
        self.it = 0
        self.n_images = 0
        self.losses_sum = 0
        self.loss_sum = 0
        self.reg_loss_sum = np.zeros(self.n_tasks)
        self.n_its = 0

    def task_normalized_training_error(self, losses):
        # transform from list to numpy
        return np.divide(losses, self.initial_losses)

    def loss_ratio(self, losses):
        with torch.no_grad():
            if self.first_epoch[0] == True:
                self.initial_losses = self.to_numpy(torch.stack(losses).detach())
            # self.task_normalized_training_error(self.to_numpy(torch.stack(losses)))
            self.losses_sum = self.task_normalized_training_error(self.to_numpy(torch.stack(losses)))
            # self.losses_sum += self.task_normalized_training_error(self.to_numpy(torch.stack(losses)))


    def mean_errors(self):
        total_loss = []
        # for i in range(self.n_tasks):
        #     # per task
        #     # mse_epoch = self.reg_loss_sum[i] / self.n_images
        #     # self.set_current_errors_string('MSETask{}'.format(i), mse_epoch)
        #     total_loss.append(self.losses_sum[i] / self.n_images)
        #     # self.set_current_errors_string('NLossTask{}'.format(i), total_loss[i])
        # self.set_current_errors(NTotalLoss=np.array(total_loss).sum()/self.n_tasks)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)
        self.set_current_errors(NTotalLoss=np.array(self.losses_sum).sum()/self.n_tasks)

    def train(self):
        self.len_data_loader = len(self.data_loader)    # check if gonna use elsewhere
        self.total_iter = 0
        for epoch in range(1, self.opt.nEpochs):
            self.epoch = epoch
            self.first_epoch = [True if epoch == 1 else False]
            self.restart_variables()
            self.data_iter = iter(self.data_loader)
            # self.pbar = tqdm(range(self.len_data_loader))
            self.pbar = range(self.len_data_loader)
            # while self.it < self.len_data_loader:
            for self.it in self.pbar:

                self.netG.train(True)

                iter_start_time = time.time()

                self.train_batch()
                self.mean_errors()

                d_time = (time.time() - iter_start_time) / self.opt.batchSize

                self.total_iter += self.opt.batchSize # change because it may be different

                # Validate
                self.evaluate(epoch)
                
                # print errors
                self.print_current_errors(epoch, d_time)

                # display errors
                self.display_current_results(epoch)


            # save checkpoint
            # self.save_checkpoint(epoch, is_best=0)


    def get_val_error(self):
        model = self.netG.train(False)
        # len_val_loader = len(self.val_loader)
        pbar_val = tqdm(self.val_loader)
        norm_losses = 0
        n_images = 0
        with torch.no_grad():
            for i, (input_cpu, target_cpu) in enumerate(pbar_val):
                input_data = input_cpu.to(self.cuda)
                output = model(input_data)
                task_loss = []
                for i_task in range(self.n_tasks):
                    target = target_cpu[:, i_task, :].to(self.cuda)
                    task_loss.append(self.regression_loss(output[:,i_task,:], target))
                
                np_task_losses = self.to_numpy(torch.stack(task_loss))

                if self.first_epoch_test == True:
                    self.val_initial_error = np_task_losses
                    self.first_epoch_test = False

                # n_images += input_cpu.shape[0]
                norm_losses += np.divide(np_task_losses, self.val_initial_error)

        return norm_losses.mean()

    def evaluate(self, epoch):
        if self.opt.validate and (epoch - 1) % self.opt.val_freq == 0:
            val_error = self.get_val_error()
            self.val_error = OrderedDict([('Val error', val_error)])
            print('Eval error is: {}'.format(val_error))

    def to_numpy(self, data):
        return data.data.cpu().numpy()

    def train_batch(self):
        pass

    def set_current_errors(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.errors.update([(key, value)])

    def set_current_errors_string(self, key, value):
        self.errors.update([(key, value)])

    def get_current_errors(self):
        return self.errors

    def get_current_errors_display(self):
        return self.errors

    def display_current_results(self, epoch):
        if self.opt.display_id > 0 and (epoch - 1) % self.opt.display_freq == 0 and self.it == (len(self.data_loader)-1):
            errors = self.get_current_errors_display()
            self.visualizer.display_errors(errors, epoch,
                                            float(self.it) / self.len_data_loader)
            if self.opt.validate and (epoch - 1) % self.opt.val_freq == 0:
                self.visualizer.display_errors(self.val_error, epoch,
                                            float(self.it) / self.len_data_loader, phase='val')

    def print_current_errors(self, epoch, d_time):
        # if self.total_iter % self.opt.print_freq == 0:
        if self.opt.display_id > 0 and (epoch - 1) % self.opt.display_freq == 0 and self.it == (len(self.data_loader)-1):
            errors = self.get_current_errors()
            message = self.visualizer.print_errors(errors, epoch, self.it,
                                            self.len_data_loader, d_time)
            print(message)
            # self.pbar.set_description(message)

class MTL_TOY_EWEIGHTS(MTL_TOY):
    def name(self):
        return 'MultiTask with Equal Weights for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        # self.set_current_visuals(input=input_data.data)

        outG = self.netG(input_data)
        self.optim.zero_grad()
        losses = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:, i_task, :].to(self.cuda)
            losses.append(self.regression_loss(outG[:,i_task,:], target))
            # self.reg_loss_sum[i_task] += self.to_numpy(losses[i_task].detach())
        
        self.loss_ratio(losses)
        with torch.no_grad():
            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        loss_error = (torch.stack(losses)).sum()
        
        loss_error.backward()
        
        self.optim.step()

        self.n_its += outG.shape[0]

    def restart_variables(self):
        MTL_TOY.restart_variables(self)

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        # for i in range(self.n_tasks):
        #     # per task
        #     mse_epoch = self.reg_loss_sum[i] / self.n_images
        #     self.set_current_errors_string('MSETask{}'.format(i), mse_epoch)

class MTL_TOY_GP_EQUAL(MTL_TOY):
    """
    1-Norm Gradient Penalty
    """
    def name(self):
        return 'MultiTask Gradient Penalty Model with equal gradients for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0
        self.regression_loss = nn.L1Loss()

    def train_batch(self):
        if self.opt.per_sample:
            self.train_batch_per_sample()
        else:
            self.train_batch_per_batch()

    def train_batch_per_batch(self): # with all real batch
        input_cpu, target_cpu = self.data_iter.next()
        batch_size = input_cpu.shape[0]
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        outG = self.netG(input_data)

        self.optim.zero_grad()
        losses = []
        norm_grad = []
        g_norm = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:,i_task,:].to(self.cuda)
            outG_task = outG[:,i_task,:]
            losses.append(self.regression_loss(outG_task, target))
            
            # print('Task id: {}'.format(i_task))
            gradients = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)
            with torch.no_grad():
                norm_grad.append(gradients.norm(2,dim=1).mean()) # mean on batch
                self.optim.zero_grad()
            
            # From WGAN-PG implementation 
            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # gradient penalty per mean batch
            # gp.append(self.alpha * ((gradients_norm-1) ** 2).mean())
            g_norm.append(gradients_norm)

        gp = self.alpha * ((g_norm[0]-g_norm[1]) ** 2).mean()

        self.loss_ratio(losses)
        # losses += gp # concatenate lists

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            # Mean value of gradients norms
            norm_grad = torch.stack(norm_grad)
            self.set_current_errors(gp=self.to_numpy(gp))

            self.grads_norm += self.to_numpy(norm_grad.mean())

            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        # losses += gp # concatenate lists
        # st()
        # print(losses[3])
        loss_error = (torch.stack(losses)).sum() + gp
        
        loss_error.backward()
        
        self.optim.step()
        self.n_its += 1 # batch size

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.n_its = 0
        self.loss_sum = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)

class MTL_TOY_GP(MTL_TOY):
    """
    1-Norm Gradient Penalty
    """
    def name(self):
        return 'MultiTask Gradient Penalty Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0
        self.regression_loss = nn.L1Loss()

    def train_batch(self):
        if self.opt.per_sample:
            self.train_batch_per_sample()
        else:
            self.train_batch_per_batch()

    def train_batch_per_batch(self): # with all real batch
        input_cpu, target_cpu = self.data_iter.next()
        batch_size = input_cpu.shape[0]
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        outG = self.netG(input_data)

        self.optim.zero_grad()
        losses = []
        norm_grad = []
        g_norm = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:,i_task,:].to(self.cuda)
            outG_task = outG[:,i_task,:]
            losses.append(self.regression_loss(outG_task, target))
            
            gradients = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)

            with torch.no_grad():
                norm_grad.append(gradients.norm(2,dim=1).mean()) # mean on batch
            
            # From WGAN-PG implementation 
            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # gradient penalty per mean batch
            # gp.append(self.alpha * ((gradients_norm-1) ** 2).mean())
            g_norm.append(gradients_norm)

        constant = 1
        gp = self.alpha * ((g_norm[0]-constant) ** 2).mean() + self.alpha * ((g_norm[1]-constant) ** 2).mean()

        self.loss_ratio(losses)
        # losses += gp # concatenate lists

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            # Mean value of gradients norms
            norm_grad = torch.stack(norm_grad)

            self.grads_norm += self.to_numpy(norm_grad.mean())

            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        # losses += gp # concatenate lists
        # st()
        # print(losses[3])
        loss_error = (torch.stack(losses)).sum() + gp
        
        loss_error.backward()
        
        self.optim.step()
        self.n_its += 1 # batch size

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.n_its = 0
        self.loss_sum = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)

class MTL_TOY_GP_PARAMS(MTL_TOY):
    """
    1-Norm Gradient Penalty
    """
    def name(self):
        return 'MultiTask Gradient Penalty Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0
        self.regression_loss = nn.L1Loss()

    def train_batch(self):
        if self.opt.per_sample:
            self.train_batch_per_sample()
        else:
            self.train_batch_per_batch()

    def train_batch_per_batch(self): # with all real batch
        input_cpu, target_cpu = self.data_iter.next()
        batch_size = input_cpu.shape[0]
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        outG = self.netG(input_data)

        self.optim.zero_grad()
        losses = []
        norm_grad = []
        g_norm = []
        gp = []
        gradients_list = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:,i_task,:].to(self.cuda)
            outG_task = outG[:,i_task,:]
            losses.append(self.regression_loss(outG_task, target))
            
            # gradients = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)

            gradients = torch.autograd.grad(losses[i_task], self.netG.parameters(), create_graph=True, retain_graph=True) # [0].view(batch_size,-1)
            
            gradients_list.append(gradients)
            with torch.no_grad():
                gradients_2 = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)
                norm_grad.append(gradients_2.norm(2,dim=1).mean()) # mean on batch
            
            # From WGAN-PG implementation 
            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            # st()

        # st()
        for i in range(len(gradients[0][:8])):
            grad1 = gradients_list[0][i].view(batch_size, -1)
            grad2 = gradients_list[1][i].view(batch_size, -1)
            gradients_norm1 = torch.sqrt(torch.sum(grad1 ** 2, dim=1) + 1e-12)
            gradients_norm2 = torch.sqrt(torch.sum(grad2 ** 2, dim=1) + 1e-12)
            g_norm.append(((gradients_norm1-gradients_norm2)**2).mean())
            # g_norm.append(((gradients_norm1)**2).mean() + ((gradients_norm2)**2).mean())
        gp.append(torch.stack(g_norm).sum())
        # constant = 1
        # gp = self.alpha * ((g_norm[0]-constant) ** 2).mean() + self.alpha * ((g_norm[1]-constant) ** 2).mean()

        self.loss_ratio(losses)
        # losses += gp # concatenate lists

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            # Mean value of gradients norms
            norm_grad = torch.stack(norm_grad)

            self.grads_norm += self.to_numpy(norm_grad.mean())
            self.gp_sum += self.to_numpy(gp[0])

            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        # losses += gp # concatenate lists
        # st()
        # print(losses[3])
        loss_error = (torch.stack(losses)).sum() + (torch.stack(gp)).sum()
        
        loss_error.backward()
        
        self.optim.step()
        self.n_its += 1 # batch size

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.n_its = 0
        self.loss_sum = 0
        self.gp_sum = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)
        mean = self.gp_sum / self.n_its
        self.set_current_errors(mean_gp=mean)

class MTL_TOY_GP_ZERO(MTL_TOY_GP):
    """
    1-Norm Gradient Penalty
    """
    def name(self):
        return 'MultiTask Gradient Penalty Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0
        self.regression_loss = nn.L1Loss(reduction='none')

    def train_batch(self):
        if self.opt.per_sample:
            self.train_batch_per_sample()
        else:
            self.train_batch_per_batch()

    def train_batch_per_batch(self): # with all real batch
        input_cpu, target_cpu = self.data_iter.next()
        batch_size = input_cpu.shape[0]
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        outG = self.netG(input_data)

        self.optim.zero_grad()
        losses = []
        norm_grad = []
        g_norm = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:,i_task,:].to(self.cuda)
            outG_task = outG[:,i_task,:]
            losses.append(self.regression_loss(outG_task, target))
        
            gradients = torch.autograd.grad(losses[0], self.netG.get_last_shared_layer().parameters(), grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)#[0].view(batch_size,-1)
            st()

            # with torch.no_grad():
            #     norm_grad.append(gradients[0].view(batch_size,-1).norm(2,dim=1).mean()) # mean on batch
        
            # From WGAN-PG implementation 
            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # gradient penalty per mean batch
            g_norm.append(gradients)

        st()
        
        # interpolate gradients
        gp = self.alpha * ((g_norm[0]) ** 2).mean() + self.alpha * ((g_norm[1]) ** 2).mean()

        self.loss_ratio(losses)

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            # for i, loss_task in enumerate(norm_grad):
            #     self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            # # Mean value of gradients norms
            # norm_grad = torch.stack(norm_grad)

            # self.grads_norm += self.to_numpy(norm_grad.mean())

            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        loss_error = (torch.stack(losses)).sum() + gp
        
        loss_error.backward()
        
        self.optim.step()
        self.n_its += 1 # batch size

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.n_its = 0
        self.loss_sum = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)

class MTL_TOY_GP_MIN(MTL_TOY_GP):
    def name(self):
        return 'MultiTask Gradient Penalty Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY_GP.initialize(self, opt)

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        # self.set_current_visuals(input=input_data.data)

        outG = self.netG(input_data)
        self.optim.zero_grad()
        losses = []
        grads = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:, i_task, :].to(self.cuda)
            losses.append(self.regression_loss(outG[:,i_task,:], target))

            self.reg_loss_sum[i_task] += self.to_numpy(losses[i_task])

            grads.append(torch.autograd.grad(losses[i_task], self.netG.parameters(), allow_unused=True, create_graph=True, retain_graph=True)[0])

        grads = grads[0] - grads[1]

        # gp = self.alpha * ((grads.norm(2, dim=1) - 1) ** 2).mean() # check norm
        gp = self.alpha * ((grads.norm(2, dim=1)) ** 2).mean() # check norm
        
        losses.append(gp)
        
        # Mean value of gradients norms
        self.grads_norm +=self.to_numpy(grads.detach().norm(2, dim=1).mean())

        loss_error = (torch.stack(losses)).sum()
        
        loss_error.backward()
        
        self.optim.step()

        self.n_images += outG[0].shape[0]

class MTL_TOY_GRADNORM(MTL_TOY):
    def name(self):
        return 'MTL GradNorm Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 0.12
        # lr_std = 2e-4
        # self.optim = torch.optim.Adam([
        #                                 {"params": self.netG.l1.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.l2.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.l3.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.l4.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.task_0.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.task_1.parameters(), "lr": lr_std*10.0},
        #                                 {"params": self.netG.omegas, "lr": lr_std}])
        # self.regression_loss = nn.MSELoss()

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        # self.set_current_visuals(input=input_data.data)

        outG = self.netG(input_data)
        self.optim.zero_grad()
        losses = []
        grads = []
        norm_grad = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:, i_task, :].to(self.cuda)
            losses.append(self.regression_loss(outG[:,i_task,:], target))

            self.reg_loss_sum[i_task] += self.to_numpy(losses[i_task])
            with torch.no_grad():
                batch_size = input_cpu.shape[0]
                gradients = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)

                # with torch.no_grad():
                #     # gradients_2 = torch.autograd.grad(losses[i_task], self.netG.parameters(), create_graph=True, retain_graph=True, allow_unused=True) # [0].view(batch_size,-1)
                #     parameters_names = [name for name, _ in self.netG.named_parameters()]
                #     parameters = [param for param in self.netG.parameters()]
                #     for i in range(len(parameters)):
                #         try:
                #             # print(parameters_names[i])
                #             # print(parameters[i].view(batch_size, -1).norm(2, dim=1))
                #             # print('[{}]: {}'.format(parameters_names[i], parameters[i].view(batch_size, -1).norm(2, dim=1)))
                #             self.set_current_errors_string('[{}]'.format(parameters_names[i]), self.to_numpy(parameters[i].view(batch_size, -1).norm(2, dim=1).mean()))
                #         except:
                #             pass
                    # for i in range(len(gradients_2)-1):
                    #     try:
                    #         print('[{}]: {}'.format(netG_parameters_names[i+1], self.to_numpy(gradients_2[i+1].norm(2,dim=1).mean())))
                    #     except:
                    #         pass
                norm_grad.append(gradients.norm(2,dim=1).mean()) # mean on batch

        self.loss_ratio(losses)
        
        # BEGIN OF GRAD NORM
        losses = torch.stack(losses)
        # GradNorm
        # if epoch == 1:
        #     with torch.no_grad():
        # initial_task_loss = self.initial_losses

        # Standard forward pass
        weighted_task_loss = torch.mul(self.netG.omegas, losses)
        loss = torch.sum(weighted_task_loss)

        self.optim.zero_grad()
        # do the backward pass to compute the gradients for the whole set of weights
        # This is equivalent to compute each \nabla_W L_i(t)
        loss.backward(retain_graph=True)
        
        # # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
        self.netG.omegas.grad.zero_()

        # # Weights of the last shared layer
        W_shared_layer = self.netG.get_last_shared_layer()
        # compute G^(i)_W(t) and r_i(t)
        # get the gradient norms for each of the tasks
        grad_norms = []
        for k in range(self.n_tasks):
            g_W_k = torch.autograd.grad(weighted_task_loss[k], W_shared_layer.parameters(), create_graph=True, only_inputs=True)[0]
            grad_norms.append(g_W_k.norm(2))

        grad_norms = torch.stack(grad_norms)

        # compute the inverse training rate r_i(t) 
        loss_ratio = losses.data.cpu().numpy() / self.initial_losses

        inverse_train_rate = loss_ratio / np.mean(loss_ratio)

        grad_mean_norm = grad_norms.mean().data.cpu().numpy()
        # compute gradnorm loss
        constant_term = torch.tensor(grad_mean_norm * (inverse_train_rate ** self.alpha), requires_grad=False).cuda()
        
        grad_norm_loss = torch.sum(torch.abs(grad_norms - constant_term))
        
        # compute the gradient for the weights
        self.netG.omegas.grad = torch.autograd.grad(grad_norm_loss, self.netG.omegas)[0]
        # END OF GRAD NORM

        self.optim.step()

        # Renormalize
        normalize_coeff = self.n_tasks / torch.sum(self.netG.omegas.data, dim=0)
        self.netG.omegas.data = self.netG.omegas.data * normalize_coeff
        with torch.no_grad():
            # self.set_current_errors(grad_loss=grad_norm_loss.data.cpu().numpy(), w_task0=self.netG.omegas[0].data.cpu().numpy(), w_task1=self.netG.omegas[1].data.cpu().numpy())
            self.set_current_errors(w_task0=self.netG.omegas[0].data.cpu().numpy(), w_task1=self.netG.omegas[1].data.cpu().numpy())
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            self.loss_sum += self.losses_sum.sum() / self.n_tasks
            norm_grad = torch.stack(norm_grad)
            self.grads_norm += self.to_numpy(norm_grad.mean())

        self.n_its += 1 # outG.shape[0]

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.loss_sum = 0
        self.n_its = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        # grads_norm_epoch = self.grads_norm / self.n_images
        # self.set_current_errors(grads_norm=grads_norm_epoch)

class MTL_TOY_GP_MAX_SCALE(MTL_TOY):
    """
    1-Norm Gradient Penalty with alpha related to max scale difference
    """
    def name(self):
        return 'MultiTask Gradient Penalty Model for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        # self.set_current_visuals(input=input_data.data)

        outG = self.netG(input_data)
        self.optim.zero_grad()
        losses = []
        grads = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:, i_task, :].to(self.cuda)
            losses.append(self.regression_loss(outG[:,i_task,:], target))

            self.reg_loss_sum[i_task] += self.to_numpy(losses[i_task])
            # self.set_current_errors_string('L1_{}'.format(i_task), self.to_numpy(losses[i_task]))
            grads.append(torch.autograd.grad(losses[i_task], self.netG.parameters(), allow_unused=True, create_graph=True, retain_graph=True)[0])

        self.loss_ratio(losses)
        with torch.no_grad():
            loss1, loss2 = self.to_numpy(losses[0]), self.to_numpy(losses[1])
            alpha1, alpha2 = np.divide(loss2, loss1), np.divide(loss1, loss2)
            self.set_current_errors(alpha1=alpha1, alpha2=alpha2)

        gp = alpha2 * ((grads[0].norm(2, dim=1)-1) ** 2).mean() + alpha1 * ((grads[1].norm(2, dim=1)-1) ** 2).mean() # 
        # st()

        with torch.no_grad():
            # Mean value of gradients norms
            grads = torch.cat(grads)
            self.grads_norm += self.to_numpy(grads.detach().norm(2, dim=1).mean())

        # gp = self.alpha * ((grads.norm(2, dim=1) - 1) ** 2).mean() # check norm
        # gp = self.alpha * ((grads.norm(2, dim=1)-1) ** 2).mean() # check norm
        losses.append(gp)
        
        loss_error = (torch.stack(losses)).sum()
        
        loss_error.backward()
        
        self.optim.step()
        self.n_images += outG.shape[0]

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_images
        self.set_current_errors(grads_norm=grads_norm_epoch)

class MTL_TOY_ALEX(MTL_TOY):
    def name(self):
        return 'MultiTask with Equal Weights for a Toy Dataset with Alex strat'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)
        self.alpha = 1.0
        self.regression_loss = nn.L1Loss()

    def train_batch(self):
        if self.opt.per_sample:
            self.train_batch_per_sample()
        else:
            self.train_batch_per_batch()

    def train_batch_per_batch(self): # with all real batch
        input_cpu, target_cpu = self.data_iter.next()
        batch_size = input_cpu.shape[0]
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        outG = self.netG(input_data)

        self.optim.zero_grad()
        losses = []
        norm_grad = []
        g_norm = []

        for i_task in range(self.n_tasks):
            target = target_cpu[:,i_task,:].to(self.cuda)
            outG_task = outG[:,i_task,:]
            losses.append(self.regression_loss(outG_task, target))
            
            with torch.no_grad():
                gradients = torch.autograd.grad(losses[i_task], input_data, grad_outputs=torch.ones(target.size()).cuda(), create_graph=True, retain_graph=True)[0].view(batch_size,-1)
                norm_grad.append(gradients.norm(2,dim=1).mean()) # mean on batch

        self.loss_ratio(losses)
        # losses += gp # concatenate lists

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            for i, loss_task in enumerate(norm_grad):
                self.set_current_errors_string('mgrad{}'.format(i), self.to_numpy(loss_task.mean()))
            # Mean value of gradients norms
            norm_grad = torch.stack(norm_grad)

            self.grads_norm += self.to_numpy(norm_grad.mean())

            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        loss_error = (torch.stack(losses)).sum()
        
        loss_error.backward()
        
        self.optim.step()
        self.n_its += 1 # batch size

    def restart_variables(self):
        MTL_TOY.restart_variables(self)
        self.grads_norm = 0
        self.n_its = 0
        self.loss_sum = 0

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        grads_norm_epoch = self.grads_norm / self.n_its
        self.set_current_errors(grads_norm=grads_norm_epoch)
        mean_loss = self.loss_sum / self.n_its
        self.set_current_errors(mean_loss=mean_loss)

class MTL_TOY_BERTRAND(MTL_TOY):

    def name(self):
        return 'MultiTask with Equal Weights for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda).requires_grad_(True)
        # self.set_current_visuals(input=input_data.data)

        losses = []
        for i_task in range(self.n_tasks):
            outG = self.netG(input_data)
            self.optim.zero_grad()

            target = target_cpu[:, i_task, :].to(self.cuda)

            loss = self.regression_loss(outG[:,i_task,:], target)
            losses.append(loss)

            with torch.no_grad():
                self.set_current_errors_string('ltask{}'.format(i_task), self.to_numpy(loss))

            loss.backward()
            
            self.optim.step()

        self.loss_ratio(losses)
        with torch.no_grad():
            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        self.n_its += 1 # outG.shape[0]

    def restart_variables(self):
        MTL_TOY.restart_variables(self)

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        # for i in range(self.n_tasks):
        #     # per task
        #     mse_epoch = self.reg_loss_sum[i] / self.n_images
        #     self.set_current_errors_string('MSETask{}'.format(i), mse_epoch)

class MTL_TOY_SPECNORM(MTL_TOY):

    def name(self):
        return 'MultiTask with Equal Weights for a Toy Dataset'

    def initialize(self, opt):
        MTL_TOY.initialize(self, opt)        
        
    def create_network(self):
        from networks.mtl_toynetwork import ToyNetworkSN
        netG = ToyNetworkSN(self.opt.n_tasks, self.opt.mtl_method)
        if self.opt.cuda:
            netG = netG.cuda()
        return netG

    def train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        
        input_data = input_cpu.to(self.cuda)
        # self.set_current_visuals(input=input_data.data)

        outG = self.netG(input_data)
        self.optim.zero_grad()
        losses = []
        for i_task in range(self.n_tasks):

            target = target_cpu[:, i_task, :].to(self.cuda)

            loss = self.regression_loss(outG[:,i_task,:], target)
            losses.append(loss)

            with torch.no_grad():
                self.set_current_errors_string('ltask{}'.format(i_task), self.to_numpy(loss))

        self.loss_ratio(losses)

        loss = torch.stack(losses).sum()
        with torch.no_grad():
            self.loss_sum += self.losses_sum.sum() / self.n_tasks

        loss.backward()
        
        self.optim.step()
        self.n_its += 1 # outG.shape[0]

    def restart_variables(self):
        MTL_TOY.restart_variables(self)

    def mean_errors(self):
        MTL_TOY.mean_errors(self)
        # for i in range(self.n_tasks):
        #     # per task
        #     mse_epoch = self.reg_loss_sum[i] / self.n_images
        #     self.set_current_errors_string('MSETask{}'.format(i), mse_epoch)