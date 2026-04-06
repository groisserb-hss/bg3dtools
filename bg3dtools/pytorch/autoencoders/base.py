"""
Base classes for variational autoencoders.

This module provides abstract base classes for building VAEs (Variational
Autoencoders) with energy-based model integration and latent space sampling.

Classes
-------
VAEDataset
    Abstract dataset class for VAE training.
BaseVAE
    Abstract base class for VAE architectures.
"""

from os.path import isfile
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from bg3dtools.pytorch.energy_models import EBM


class VAEDataset(Dataset, ABC):
        def __init__(self, *args, **kwards) -> None:
            super(VAEDataset, self).__init__()
            self.names = []

        def __len__(self):
            return len(self.names)

        @abstractmethod
        def manual_getitem(self, idx, max_batch=None) -> list:
            """
            This function gets called directly, without a DataLoader object, and produces a single sample
            output is consumed directly by BaseVAE.encode (e.g. must be batched)
            """

            raise NotImplementedError

        def sample(self, N: int, max_batch=1):
            assert N <= len(self), 'not enough samples'

            indices = np.random.choice(len(self), N, replace=False)
            samples = [self.manual_getitem(idx, max_batch=max_batch) for idx in indices]
            num_arrays = len(samples[0])
            samples = [np.stack([s[i] for s in samples], axis=0) for i in range(num_arrays)]
            return samples


class BaseVAE(EBM):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

        self.loss_args = {'M_N': 1.}
        self.device = torch.device('cpu')

    @abstractmethod
    def encode(self, input: list, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def decode(self, input: list, **kwargs):
        raise NotImplementedError

    def _batch_pdist(self, x):
        """
        Compute the pairwise distance matrix for a batch of vectors
        can be useful for enforcing similarity in distance across raw/latent spaces
        """
        B = x.shape[0]
        x = x.view(B, 1, -1)
        x = x.repeat(1, B, 1)
        xT = x.permute(1, 0, 2)
        dist = torch.norm(x - xT, p=2, dim=-1)
        dist = dist / torch.mean(dist)
        return dist

    def generate(self, x: list, **kwargs):
        """
        Given an input x, returns the reconstruction
        :param x: (Tensor) [B x C]
        :return: (Tensor) [B x C]
        """
        z = self.encode(x)[0]
        return self.decode(z)

    def forward(self, inputs: list, **kwargs) -> list[torch.Tensor]:
        """
        Given an input, returns the reconstruction after adding noise
        :param inputs: [raw]
        :return: [raw, recon, mu, log_var]
        """
        # default forward pass has only one input
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        recon = self.decode([z])
        return inputs + [recon, mu, log_var]

    def loss_function(self, tensor_list: list) -> (Tensor, float, float):
        """
        Compute VAE loss for a batch
        """
        x, x_, mu, log_var = tensor_list

        kld_weight = self.loss_args['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = torch.nn.functional.mse_loss(x, x_)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return loss, recons_loss.item(), kld_weight * kld_loss.item()

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def latent_fit(self, arrays: list, iters=200, lr=0.05, noise_decay=0.98, num_samples=100):
        """
        Optimize latent code for a given input
        :param arrays: codes will be prepended and passed to optimize_samples
        """
        num_cases = max([len(a) for a in arrays])
        for a in arrays:
            assert len(a) == 1 or len(a) == num_cases, 'batch dimension mismatch'

        # encode data as initialization
        with torch.no_grad():
            tensors = [torch.tensor(d, device=self.device, requires_grad=False) for d in arrays]
            mu, log_var = self.encode(tensors)

            # broadcast if necessary
            if len(mu) < num_samples:
                scale = np.ceil(num_samples / len(mu)).astype(int)
                mu = torch.tile(mu, (scale, 1))
                log_var = torch.tile(log_var, (scale, 1))
            mu, log_var = mu[:num_samples], log_var[:num_samples]

            codes = self.reparameterize(mu, log_var)
            codes = codes.detach().cpu().numpy()

        coherence, z_opt = self.optimize_samples([codes] + arrays, iters=iters, lr=lr, noise_decay=noise_decay)[:2]
        return coherence, z_opt, mu.detach().cpu().numpy()


def train_model(model: BaseVAE, model_file, train_dataset, test_dataset,
                batch_size=1, num_epochs=5000, lr=0.01, chkpts=[]):
    import numpy as np
    from torch.utils.data import DataLoader

    assert model_file[-4:] == '.pth'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.to(model.device)
    train_loss, test_loss = model.fit(train_loader, test_loader,
                                      num_epochs=num_epochs, learning_rate=lr, model_file=model_file, chkpts=chkpts)

    model.to(torch.device('cpu'))
    torch.save(model.state_dict(), model_file)
    np.savez(model_file.replace('.pth', '_loss.npz'), train_loss=train_loss, test_loss=test_loss)


def test_model(model: BaseVAE, eval_file, test_dataset: VAEDataset):
    import numpy as np
    from umap import UMAP
    noise_decay, iters, num_samples, num_cases = 0.99, 1024, 24, 8

    # evaluate model

    if isfile(eval_file):
        result = np.load(eval_file)
        coherence, encoded, optimized = result['coherence'], result['encoded'], result['optimized']
    else:
        # optimize for latent code from each subject (use all sequences)
        model.to(model.device)
        nP = len(test_dataset)
        coherence = np.zeros(nP, dtype=np.float32)
        encoded = np.nan * np.zeros([nP, num_samples, model.latent_dim], dtype=np.float32)
        optimized = np.nan * np.zeros([nP, num_samples, model.latent_dim], dtype=np.float32)
        for ss in range(len(test_dataset)):
            print('subject %d/%d' % (ss+1, len(test_dataset)))
            data = test_dataset.manual_getitem(ss, max_batch=num_cases)
            coherence[ss], optimized[ss], encoded[ss] = model.latent_fit(data,
                                                noise_decay=noise_decay, iters=iters, num_samples=num_samples)

        np.savez(eval_file, names=test_dataset.names, coherence=coherence, optimized=optimized, encoded=encoded)

    opt_mean = np.nanmean(optimized, axis=1)
    # project into 3D for visualization
    umap_3d = UMAP(n_components=3, init='random', random_state=0)
    proj_3d = umap_3d.fit_transform(opt_mean)

    print('coherence: %.3f' % np.mean(coherence))
    print('latent code mean:')
    print(np.mean(opt_mean, axis=0))
    print('latent code std:')
    print(np.std(opt_mean, axis=0))

    return proj_3d, optimized

