# a (mostly) single-file implementation
# based on https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating
# and https://colab.research.google.com/drive/15S4ISFVMQtfc0FPi29HRaX03dWxL65zx?usp=sharing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from dataclasses import dataclass
from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class Config():
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42

    # mlp config
    input_dim: int = 5
    hidden_dim: int = 2
    output_dim: int = 5

    # dataset and training config
    num_instances: int = 4 # number of models to train
    num_steps: int = 10_000

    batch_size: int = 1024
    lr: float = 1e-3


class Bottleneck_Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.W = nn.Parameter(torch.empty((cfg.input_dim, cfg.hidden_dim)))
        self.b = nn.Parameter(torch.zeros(cfg.input_dim))
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.W)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = x @ self.W # [batch_size, hidden_dim]
        x = x @ self.W.T + self.b # [batch_size, input_dim]
        return self.relu(x)


def gen_artificial_data(cfg, feature_prob):
    # generates synthetic dataset of shape [batch_size, input_dim]
    #
    # feature_prob: scalar
    #       each dimension in an input example represents a feature
    #       in a given train example, each feature shows up randomly with probability p
    #       if it is present, it has a value in [0, 1)
    #       * higher feature probability = lower sparsity
    #       * intuition/expected behavior - less superposition/interference used to represent high-prob features

    device = cfg.device
    
    seeds = torch.rand((cfg.batch_size, cfg.input_dim))
    dataset = torch.where(seeds <= feature_prob, torch.rand(1), 0).to(device)

    return dataset


def calc_reconstruction_loss(X_hat, X, importances):
    # basically just MSE
    # X_hat, X: [batch_size, input_dim]
    # importances: [input_dim]

    return (importances * ((X_hat - X) ** 2)).mean()


def train_bottleneck_mlp(cfg, model, feature_prob, importances):
    # one epoch over the entire trainset
    # no need for valset, since we don't care about real world performance here
    #
    # importances: [input_dim]

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_steps)

    with tqdm(range(cfg.num_steps)) as pbar:
        for step in pbar:
            X = gen_artificial_data(cfg, feature_prob) # [batch_size, input_dim]
            X_hat = model(X)

            loss = calc_reconstruction_loss(X_hat, X, importances)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=loss.item())
            

def plot_learned_features(cfg, models, feature_probs):
    fig = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[f'feat. prob. = {feature_prob:.3f}' for feature_prob in feature_probs]
    )

    for idx, model in enumerate(models):
        feats = model.W.detach().cpu().numpy()

        for vec in feats:
            fig.add_trace(go.Scatter(
                x = (0, vec[0]),
                y = (0, vec[1]),
                mode='lines+markers',
                marker=dict(color='black', size=10),
            ), row=1, col=idx+1)

            fig.update_xaxes(range=[-1.5, 1.5], row=1, col=idx+1)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=idx+1)

    fig.update_layout(
        width=400*len(models),
        height=400,
        title_text='5 features represented in 2D space'
    )
    fig.show()


if __name__ == '__main__':
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device

    # try num_instances models with varying feature probabilities
    # mimicking a 'real' training setting where some features show up more often than others
    # each feature has the same probability and we vary this across models
    feature_probs = (50 ** -torch.linspace(0, 1, cfg.num_instances))

    # in a 'real' training setting, some features are more important than others
    # we'll mimic this with an 'importance' score that will go into the loss fn
    importances = (0.9 ** torch.arange(cfg.input_dim)).to(device)

    models = [Bottleneck_Model(cfg).to(device) for _ in range(cfg.num_instances)]

    # this is slow but it makes the tensor dims less cluttered
    for idx, model in enumerate(models):
        train_bottleneck_mlp(cfg, model, feature_probs[idx], importances)
    
    plot_learned_features(cfg, models, feature_probs)
