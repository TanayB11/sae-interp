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
    num_instances: int = 5 # number of models to train
    num_examples: int = 10_000_000 # trainset size

    batch_size: int = 1024
    lr: float = 1e-3


class Bottleneck_MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.input_dim)
        self.relu = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.fc1(x)
        x = self.fc2(x)
        return self.relu(x)


def gen_artificial_data(cfg, feature_probs):
    # generates synthetic dataset of shape [num_instances, num_examples, input_dim]
    #
    # feature_probs: [1, input_dim]
    #       each dimension in an input example represents a feature
    #       in a given train example, each feature shows up randomly with probability p
    #       if it is present, it has a value in [0, 1)
    #       * higher feature probability = lower sparsity
    #       * intuition/expected behavior - less superposition used to represent high-prob features

    device = cfg.device
    
    dataset = torch.rand((cfg.num_instances, cfg.num_examples, cfg.input_dim))
    dataset = dataset.where(dataset <= feature_probs, 0).to(device)

    return dataset


def calc_reconstruction_loss(X_hat, X):
    # basically just MSE
    # X_hat, X: [batch_size, input_dim]
    return ((X_hat - X) ** 2).mean()


def train_bottleneck_mlp(cfg, model, train_loader):
    # one epoch over the entire trainset
    # no need for valset, since we don't care about real world performance here

    num_steps = len(train_loader)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps)

    with tqdm(train_loader) as pbar:
        for step, X in enumerate(pbar):
            # X: [batch_size, input_dim]

            X = X[0].to(device) # X comes from dataloader as a list
            X_hat = model(X)

            loss = calc_reconstruction_loss(X_hat, X)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

            pbar.set_postfix(loss=loss.item())
            
def plot_learned_features(cfg, models, feature_probs):
    fig = make_subplots(
        rows=1,
        cols=len(models),
        subplot_titles=[f'feat. prob. = {feature_prob:.4f}' for feature_prob in feature_probs]
    )

    # we can train all the models together, but i think the code
    # is more intuitive to understand with a loop over all models
    for idx, model in enumerate(models):
        feats = model.fc1.weight.T.detach().cpu().numpy()

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
    feature_probs = (50 ** -torch.linspace(0, 1, cfg.num_instances))
    data = gen_artificial_data(cfg, feature_probs) # [num_instances, num_examples, input_dim]

    models = [Bottleneck_MLP(cfg).to(device) for _ in range(cfg.num_instances)]
    for idx, model in enumerate(models):
        trainset = TensorDataset(data[idx]) # each element is a 1-tuple
        train_loader = DataLoader(
            trainset, 
            batch_size=cfg.batch_size, 
        )
        train_bottleneck_mlp(cfg, model, train_loader)
    
    plot_learned_features(cfg, models, feature_probs)