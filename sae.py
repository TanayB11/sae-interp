import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoModelForCausalLM
from utils import get_tokenizer_and_loaders

from dataclasses import dataclass
from tqdm import tqdm


class SAE(nn.Module):
    # based on https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder
    # we choose to train an SAE on the MLP activations in the 2nd layer

    def __init__(self, mlp_dim, hidden_ratio):
        super().__init__()

        self.encoder = nn.Linear(mlp_dim, mlp_dim * hidden_ratio)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(mlp_dim * hidden_ratio, mlp_dim)

        self.bias_encoder = nn.Parameter(torch.empty((mlp_dim * hidden_ratio)))
        self.bias_decoder = nn.Parameter(torch.empty((mlp_dim)))

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.normal_(self.bias_encoder)
        nn.init.normal_(self.bias_decoder)
    
    def forward(self, x):
        x_bar = x - self.bias_decoder
        f = self.relu(x_bar)
        x_hat = self.decoder(f) + self.bias_decoder

        # f contains our learned features (hidden state)
        return x_hat, f
    
def sae_loss(x, x_hat, f, lambd):
    # SAE reconstruction loss w/ sparsity penalty
    # x (MLP activations):    [batch_size, mlp_dim]
    # x_hat (reconstruction): [batch_size, mlp_dim]
    # f (SAE activations):    [batch_size resid_dim * hidden_ratio]
    # lambd: L1 regularization (for sparsity)

    penalized_l2_squared_diff = (x - x_hat).norm(dim=1) ** 2 + lambd * f.norm(p=1)
    return penalized_l2_squared_diff.mean()

def train_sae(cfg, sae, model, train_loader, val_loader):
    device = cfg.device

    mlp_activations = []
    def layer2_mlp_hook(layer, args, output):
        # output: [batch_size, 1, mlp_hidden_dim]
        mlp_activations.append(output[:, 0, :])

    # we choose to train an SAE on the MLP activations in the 2nd layer
    model.transformer.h[1].mlp.c_fc.register_forward_hook(layer2_mlp_hook)
    model.eval() # don't need gradients from the transformer

    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    for epoch in tqdm(range(cfg.num_epochs)):
        with tqdm(train_loader) as pbar:
            sae.train()

            for step, input_tokens in enumerate(pbar):
                mlp_activations = []
                model_output = model.generate(**input_tokens, max_length=cfg.max_ctx_len, num_beams=1)

                # this is our input data to the SAE
                #   total_output_len is the number of output tokens across the entire batch
                #   which is the batch size for the SAE training data
                mlp_activations = torch.cat(mlp_activations) # [total_output_len-1, mlp_dim]

                sae_reconstruction, sae_hidden_state = sae(mlp_activations)

                loss = sae_loss(mlp_activations, sae_reconstruction, sae_hidden_state, cfg.sae_l1_penalty)
                opt.zero_grad()
                loss.backward()
                opt.step()

                pbar.set_postfix(loss=loss.item())
                
            sae.eval()
            losses = []
            for input_tokens in val_loader:
                mlp_activations = []
                model_output = model.generate(**input_tokens, max_length=cfg.max_ctx_len, num_beams=1)

                mlp_activations = torch.cat(mlp_activations)
                sae_reconstruction, sae_hidden_state = sae(mlp_activations)

                losses.append(sae_loss(mlp_activations, sae_reconstruction, sae_hidden_state, cfg.sae_l1_penalty).item())
            
            avg_loss = torch.tensor(losses).mean()
            print(f'Epoch {epoch} validation: loss {avg_loss:.4f}')


@dataclass
class Config:
    device: str = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    seed: int = 42

    trainset_size: int = 10000
    batch_size: int = 32 # increase on better hardware
    num_epochs: int = 50
    lr: float = 3e-4

    mlp_dim: int = 4096
    max_ctx_len: int = 2048
    sae_hidden_ratio: int = 1 # increase to ~256 on better hardware
    sae_l1_penalty: float = 0.008  # definitely needs tuning


if __name__ == '__main__':
    cfg = Config()

    device = cfg.device
    print(f'Running on device {device}')

    sae = SAE(cfg.mlp_dim, cfg.sae_hidden_ratio).to(device)
    model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-2Layers-33M').to(device)

    tokenizer, train_loader, val_loader = get_tokenizer_and_loaders(cfg)
    model.generation_config.pad_token_id = tokenizer.eos_token_id # suppress a huggingface warning :p

    train_sae(cfg, sae, model, train_loader, val_loader)