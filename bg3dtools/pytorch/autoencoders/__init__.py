from .base import VAEDataset, BaseVAE, train_model, test_model
from .vanilla_1d import Vanilla1DVAE
from .vanilla_2d import Vanilla2DVAE
from .cvae_2d import Conditional2DVAE
from .cat_vae_2d import Categorical2DVAE

__all__ = [
    "VAEDataset", "BaseVAE", "train_model", "test_model",
    "Vanilla1DVAE", "Vanilla2DVAE", "Conditional2DVAE", "Categorical2DVAE",
    "VAE2D", "CVAE2D", "CATVAE2D", "vae_models",
]

# Aliases
VAE2D = Vanilla2DVAE
CVAE2D = Conditional2DVAE
CATVAE2D = Categorical2DVAE

vae_models = {'VanillaVAE':Vanilla2DVAE,
              'ConditionalVAE':Conditional2DVAE,
              'CategoricalVAE':Categorical2DVAE}