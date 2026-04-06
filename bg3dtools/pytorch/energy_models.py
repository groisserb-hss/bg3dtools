"""
Energy-based models with Langevin dynamics sampling.

This module provides base classes for energy-based models (EBMs) with
Stochastic Gradient Langevin Dynamics (SGLD) for sampling from learned
energy landscapes.

Classes
-------
SGLD
    Stochastic Gradient Langevin Dynamics optimizer.
EBM
    Abstract base class for energy-based models.
"""

from abc import ABC, abstractmethod
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from torch.optim.optimizer import Optimizer
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SGLD(Optimizer):
    """Stochastic Gradient Langevin Dynamics optimizer.
    """

    def __init__(self, params, lr, jitter=2.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(SGLD, self).__init__(params, defaults)
        self.jitter = jitter

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = torch.tensor(self.jitter * np.sqrt(group['lr']), device=p.data.device)
                # use new constructor
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1) * noise_std
                p.data.add_(noise)

        return loss


class EBM(ABC, nn.Module):
    def __init__(self, device='cpu'):
        super(EBM, self).__init__()
        self.device = torch.device(device)
        self.train_iter = 0  # for debugging

    @abstractmethod
    def paired_energy(self, data: list) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_function(self, data: list) -> torch.Tensor:
        pass

    def fit(self, train_loader: DataLoader, test_loader: DataLoader = None,
            num_epochs=5000, learning_rate=0.005, model_file='model.pth', chkpts=[]):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        step_size = num_epochs // 10
        scheduler = ReduceLROnPlateau(optimizer, patience=step_size, factor=0.5, threshold=0.001,
                                      threshold_mode='rel', min_lr=learning_rate/100)

        train_loss = np.zeros(num_epochs)
        test_loss = np.nan * np.zeros(num_epochs)
        progress = tqdm(range(num_epochs), desc='')
        for ii in progress:
            self.train_iter = ii
            tr_loss, reg_loss = self.train_epoch_(train_loader, optimizer)
            train_loss[ii] = np.sqrt(tr_loss)
            metric = tr_loss

            if test_loader is not None:
                te_loss = self.test_epoch_(test_loader)
                test_loss[ii] = np.sqrt(te_loss)
                metric = te_loss

            scheduler.step(metric)
            lr = optimizer.param_groups[0]['lr']
            tr_loss, te_loss = float(train_loss[ii]), float(test_loss[ii])
            te_mean = np.nan if ii < 100 else float(test_loss[ii-100:ii].mean())
            s = '%.5f :  %.3f (+%.3f) - %.3f (%.3f)' % (lr, tr_loss, np.sqrt(reg_loss), te_loss, te_mean)
            progress.set_postfix_str(s)

            if ii in chkpts:
                self.to(torch.device('cpu'))
                torch.save(self.state_dict(), model_file.replace('.pth', '_%d.pth' % ii))
                np.savez(model_file.replace('.pth', '_%d.npz' % ii),
                         train_loss=train_loss, test_loss=test_loss)
                self.to(self.device)

            if ii % 10 == 0:
                # clear memory
                gc.collect()
                torch.cuda.empty_cache()

        return train_loss, test_loss

    def train_epoch_(self, train_loader: DataLoader, optimizer: Optimizer):
        self.train()
        recon_loss, reg_loss, num_samples = 0, 0, 0
        for data in train_loader:
            data = [d.to(self.device) for d in data]
            optimizer.zero_grad()
            outputs = self.forward(data)
            loss, data_loss, regularization = self.loss_function(outputs)
            if loss is None:
                continue
            loss.backward()

            torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)

            optimizer.step()
            n = len(data[0])
            recon_loss += data_loss.item() * n
            reg_loss += regularization.item() * n
            num_samples += n
        return recon_loss / num_samples, reg_loss / num_samples

    def test_epoch_(self, test_loader):
        self.eval()
        recon_loss = 0
        with torch.no_grad():
            for data in test_loader:
                data = [d.to(self.device) for d in data]
                outputs = self.forward(data)
                loss, data_loss, kld_loss = self.loss_function(outputs)
                recon_loss += data_loss.item() * len(data[0])
        return recon_loss / len(test_loader.dataset)

    def langevin_sample(self, tensors: list, iters=100, lr=0.05,
                        noise_decay=1.0, init_step=2.0) -> (torch.Tensor, torch.Tensor):
        """
        stochastic gradient descent over the energy function
        """
        self.eval()
        self.requires_grad_(False)

        params = [t for t in tensors if t.requires_grad]
        static = [t for t in tensors if not t.requires_grad]
        num_samples = len(params[0])
        num_cases = max([len(t) for t in static])

        # manually broadcast all static tensors
        for i, t in enumerate(tensors):
            if t.requires_grad: continue
            dims = [1] * len(t.shape)
            dims[0] = 1 if len(t) == 1 else num_samples
            tensors[i] = torch.tile(t, dims)

        optimizer = SGLD(params, jitter=init_step, lr=lr)

        progress = tqdm(range(iters), desc='')
        for i in progress:
            optimizer.jitter *= noise_decay

            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()

                new_tensors = [t for t in tensors]
                # broadcast parameters
                for i, t in enumerate(new_tensors):
                    if not t.requires_grad: continue
                    dims = [1] * len(t.shape)
                    dims[0] = num_cases if len(t) > 1 else 1
                    new_tensors[i] = torch.tile(t, dims)

                energy = self.paired_energy(new_tensors)

                if energy.requires_grad:
                    energy.backward()
                return energy

            optimizer.step(closure)
            energy = closure()
            progress.set_postfix_str('%.3f' % energy.item())

        return tensors

    def optimize_samples(self, arrays: [np.ndarray, np.ndarray], optimize=(True, False),
                        iters=200, lr=0.05, noise_decay=0.98, return_point=True) -> tuple:
        """class-specific wrapper for langevin_sample
        default is for single-array input and output
        arrays must be batched, but if either has a single element it will be broadcast
        : param arrays: [x, y]
            x = inputs to the model (latent encoding)
            y = outputs of the model (data space)
        : param optimize: (bool, bool)
        returns mean and variance of all optimized arrays
        """
        assert len(arrays) == 2
        x = arrays[0]
        y = arrays[1]

        x_t = torch.tensor(x, device=self.device, dtype=torch.float32, requires_grad=optimize[0])
        y_t = torch.tensor(y, device=self.device, dtype=torch.float32, requires_grad=optimize[1])

        x_opt, y_opt = self.langevin_sample([x_t, y_t], iters=iters, lr=lr, noise_decay=noise_decay)

        x_mean = torch.mean(x_opt, dim=0, keepdim=True)
        coherence = self.paired_energy([x_mean, y_opt]).item()

        # convert back to numpy
        x_opt = x_opt.detach().cpu().numpy()
        y_opt = y_opt.detach().cpu().numpy()

        return coherence, x_opt, y_opt
