from .mtl_train import MultiTaskGen
from ipdb import set_trace as st
import numpy as np
import torch

class MTL_GP(MultiTaskGen):
    def name(self):
        return 'MTL Gradient Penalty Large datasets'

    def initialize(self, opt):
        opt.normalization = 'layer_normalization'
        MultiTaskGen.initialize(self, opt)

        # add here other initialization
        self.alpha = 1.0

    def _train_batch(self):
        input_cpu, target_cpu = self.data_iter.next()
        # get batch size for each batch
        batch_size = input_cpu.shape[0]
        # for int_it in range(batch_input_cpu.shape[0]):
        # input_cpu = batch_input_cpu[int_it, :,:,:].unsqueeze(0) # is dimension correct
        input_data = input_cpu.to(self.cuda)
        input_data.requires_grad = True
        constant = 1
        self.set_current_visuals(input=input_data.data)

        outG = self.netG.forward(input_data)
        
        self.optimG.zero_grad()
        losses = []
        grads = []
        gp = []
        norm_grad = []
        g_norm = []
        for i_task, task in enumerate(self.opt.tasks):
            # check here
            target = target_cpu[i_task].to(self.cuda)
            # target = target[int_it].unsqueeze(0).to(self.cuda)
            if task == 'semantics':
                losses.append(self.cross_entropy(outG[i_task], target))
                self.get_errors_semantics(target, outG[i_task], n_classes=self.opt.outputs_nc[i_task])
                # losses.append(self.train_semantics(target, outG[i_task]))
            elif task == 'depth':
                losses.append(self.criterion_reg(target, outG[i_task]))
                self.get_errors_regression(target, outG[i_task])
                
            gradients = torch.autograd.grad(losses[i_task], input_data, create_graph=True, retain_graph=True, allow_unused=False)[0].view(batch_size,-1) # check allow used

            with torch.no_grad():
                norm_grad.append(self.to_numpy(gradients.norm(2,dim=1).mean())) # mean on batch
            
            # From WGAN-PG implementation 
            # Derivatives of the gradient close to 0 can cause problems because of
            # the square root, so manually calculate norm and add epsilon
            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            # gradient penalty per mean batch
            # gp.append(self.alpha * ((gradients_norm-1) ** 2).mean())
            # g_norm.append(gradients_norm)

            gp.append(self.alpha * ((gradients_norm-1) ** 2).mean())

        # concatenate lists
        losses += gp

        self.loss_error = sum(losses)

        self.loss_error.backward()
        self.optimG.step()

        self.n_iterations += 1

        with torch.no_grad():
            # show each loss
            for i, loss_task in enumerate(losses):
                self.set_current_errors_string('loss{}'.format(i), self.to_numpy(loss_task))
            self.norm_grad_sum += np.array(norm_grad).mean()
            norm_grad = self.norm_grad_sum / self.n_iterations
            self.set_current_errors(norm_grad=norm_grad)

    def restart_variables(self):
        MultiTaskGen.restart_variables(self)
        self.norm_grad_sum = 0

    