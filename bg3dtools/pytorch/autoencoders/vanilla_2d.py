"""
Vanilla 2D variational autoencoder.

Convolutional VAE for image data with configurable encoder/decoder
architectures and latent dimension.
"""

import torch
from torch import Tensor
from ..autoencoders import BaseVAE
from torch import nn
from torch.nn import functional as F


class Vanilla2DVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 latent_dim: int,
                 img_size: list,
                 conv_channels: list = None,
                 kernel_sizes: list = None,
                 hidden_dims: list = None,
                 limit: float = 2,
                 **kwargs) -> None:
        super(Vanilla2DVAE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.raw_size = img_size[:]
        self.limit = limit

        if conv_channels is None:
            conv_channels = [32, 64, 128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3, 3]
        if hidden_dims is None:
            hidden_dims = []

        # Image reduction
        modules = []
        for c, k in zip(conv_channels, kernel_sizes):
            assert k % 2 == 1
            p = (k-1) // 2
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=c, kernel_size=k, stride=2, padding=p),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU())
            )
            in_channels = c
        H = img_size[0] // (2 ** len(conv_channels))
        W = img_size[1] // (2 ** len(conv_channels))

        self.img_encoder = nn.Sequential(*modules)
        self.reduced_size = (H, W)

        # build MLP to latent vectors
        modules_mu, modules_var = [], []
        in_channels = H * W * in_channels
        for l in (hidden_dims + [latent_dim]):
            modules_mu.append(nn.Linear(in_channels, l))
            modules_var.append(nn.Linear(in_channels, l))
            in_channels = l

        self.fc_mu = nn.Sequential(*modules_mu)
        self.fc_var = nn.Sequential(*modules_var)

        # MLP from latent vector back to pooled image
        hidden_dims = hidden_dims[::-1]
        conv_channels = conv_channels[::-1]
        modules_out = []
        for l in hidden_dims + [conv_channels[0] * H * W]:
            modules_out.append(nn.Linear(in_channels, l))
            in_channels = l
        self.decoder_input = nn.Sequential(*modules_out)

        # image expansion
        modules = []
        conv_channels += [conv_channels[-1]]
        steps = len(conv_channels) - 1
        for i in range(steps):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(conv_channels[i],
                        conv_channels[i + 1], kernel_size=3,
                        stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(conv_channels[i + 1]),
                    nn.LeakyReLU())
            )
        self.img_decoder = nn.Sequential(*modules)

        self.final_layer = nn.Conv2d(conv_channels[-1], out_channels=out_channels, kernel_size=3, padding=1)

    def encode(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.img_encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        batch_size = result.shape[0]
        result = result.view(batch_size, -1, self.reduced_size[0], self.reduced_size[1])
        result = self.img_decoder(result)
        result = self.final_layer(result)
        result = torch.nn.functional.tanh(result / (self.limit * 10)) * (self.limit * 10)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
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

    def forward(self, input: Tensor, jitter=True) -> list[Tensor]:
        mu, log_var = self.encode(input)
        if jitter:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD': kld_weight * kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, jitter=True) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, jitter)[0]