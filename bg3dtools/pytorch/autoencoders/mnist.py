"""
MNIST VAE training script.

Example training loop for the Vanilla1DVAE on MNIST digit data.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import torch
from vanilla_1d import Vanilla1DVAE
from torch.nn import functional as F
import torch.nn as nn


# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Hyperparameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 10

# MNIST dataset
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # Mean vector
        self.fc22 = nn.Linear(400, 20)  # Log variance vector

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

# Model
model = Vanilla1DVAE(data_dim=784, latent_dim=20, hidden_dims=[400]).to(device)
# model = VAE().to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def vae_loss(recon_x, x, mu, logvar) -> torch.Tensor:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, input, mu, logvar = model(data.reshape(-1, 784))
        recon_batch = torch.sigmoid(recon_batch)
        loss = vae_loss(recon_batch, input, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader.dataset)}')

# Testing
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        recon, input, mu, logvar = model(data.reshape(-1, 784))
        recon = F.sigmoid(recon)
        test_loss += vae_loss(recon, input, mu, logvar).item()

print(f'Test Loss: {test_loss / len(test_loader.dataset)}')