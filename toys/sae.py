# a (mostly) single-file implementation
# run the main code on sae_demos.ipynb!
# based on https://transformer-circuits.pub/2023/monosemantic-features/index.html
# and https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx?usp=sharing

import torch
import torch.nn as nn
import torch.optim as optim

from bottleneck_models import Bottleneck_Model, gen_artificial_data, train_bottleneck_model, plot_learned_features

from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Config():
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42

    # bottleneck model config
    input_dim: int = 5  # number of actual features in dataset
    hidden_dim: int = 2
    output_dim: int = 5

    # sae config
    sae_hidden_dim: int = input_dim # should be at least num_features
    sae_l1_penalty: float = 0.008   # definitely needs tuning

    # dataset and training config
    num_steps: int = 10_000 # bottleneck model
    batch_size: int = 1024  # bottleneck model
    lr: float = 1e-3


class SAE(nn.Module):
    # based on https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder

    def __init__(self, cfg):
        super().__init__()

        self.encoder = nn.Parameter(torch.empty(cfg.hidden_dim, cfg.sae_hidden_dim))
        self.relu = nn.ReLU()
        self.decoder = nn.Parameter(torch.empty(cfg.sae_hidden_dim, cfg.hidden_dim))

        self.bias_encoder = nn.Parameter(torch.zeros((cfg.sae_hidden_dim)))
        self.bias_decoder = nn.Parameter(torch.zeros((cfg.hidden_dim)))

        nn.init.kaiming_uniform_(self.encoder)
        nn.init.kaiming_uniform_(self.decoder)
    
    def forward(self, x):
        # x is hidden state from model: [batch_size, hidden_dim]
        x_bar = x - self.bias_decoder # [batch_size, hidden_dim]

        # contains our learned features
        sae_hidden = self.relu(x_bar @ self.encoder + self.bias_encoder) # [batch_size, sae_hidden_dim]

        reconstruction = sae_hidden @ self.decoder + self.bias_decoder # [batch_size, hidden_dim]
        return reconstruction, sae_hidden


def sae_loss(model_acts, reconstruction, sae_hidden, l1_penalty):
    # SAE reconstruction loss w/ sparsity penalty
    # model_acts:          [batch_size, hidden_dim]
    # reconstruction:      [batch_size, hidden_dim]
    # f (SAE activations): [batch_size, sae_hidden_dim]

    l2_squared_diff = (model_acts - reconstruction).norm(dim=1) ** 2
    penalized_l2_squared_diff = l2_squared_diff + l1_penalty * sae_hidden.norm(p=1)
    return penalized_l2_squared_diff.mean()


def train_sae(cfg, sae, model, feature_prob):
    device = cfg.device

    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_steps)

    with tqdm(range(cfg.num_steps)) as pbar:
        for step in pbar:
            # per the paper, ensure SAE decoder weight columns have unit norm
            sae.decoder.data = sae.decoder.data / sae.decoder.data.norm(dim=1, keepdim=True)

            with torch.inference_mode():
                batch = gen_artificial_data(cfg, feature_prob) # [batch_size, input_dim]
                model_acts = batch @ model.W # [batch_size, hidden_dim]

            reconstruction, sae_hidden = sae(model_acts)
            loss = sae_loss(model_acts, reconstruction, sae_hidden, cfg.sae_l1_penalty)

            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=loss.item())

