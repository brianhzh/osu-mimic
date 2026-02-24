import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AimGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 17,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)

        out = self.fc1(gru_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)

        return out, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


class AimMDN(nn.Module):
    def __init__(
        self,
        input_size: int = 17,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_mixtures: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc_trunk = nn.Linear(hidden_size, 128)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        self.fc_pi = nn.Linear(128, num_mixtures)
        self.fc_mu = nn.Linear(128, num_mixtures * 2)
        self.fc_sigma = nn.Linear(128, num_mixtures * 2)

    def forward(self, x, hidden=None):
        batch, seq, _ = x.shape
        K = self.num_mixtures

        gru_out, hidden = self.gru(x, hidden)

        trunk = self.relu(self.fc_trunk(gru_out))
        trunk = self.dropout(trunk)

        pi = torch.softmax(self.fc_pi(trunk), dim=-1)
        mu = self.fc_mu(trunk).view(batch, seq, K, 2)
        sigma = F.softplus(self.fc_sigma(trunk)).view(batch, seq, K, 2) + 1e-4

        return pi, mu, sigma, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)


def get_mean_prediction(pi, mu):
    return torch.sum(pi.unsqueeze(-1) * mu, dim=2)


def mdn_loss(pi, mu, sigma, target):
    target_expanded = target.unsqueeze(2).expand_as(mu)

    log_normal = (
        -0.5 * np.log(2 * np.pi)
        - torch.log(sigma)
        - 0.5 * ((target_expanded - mu) / sigma) ** 2
    )
    log_component = log_normal.sum(dim=-1)

    log_pi = torch.log(pi + 1e-8)
    log_weighted = log_pi + log_component

    log_likelihood = torch.logsumexp(log_weighted, dim=-1)
    return -log_likelihood.mean()


def sample_from_mdn(pi, mu, sigma, temperature=1.0):
    if temperature < 1e-6:
        k = torch.argmax(pi[0, 0])
        return mu[0:1, 0:1, k:k+1, :].squeeze(2)

    log_pi = torch.log(pi + 1e-8) / temperature
    pi_temp = torch.softmax(log_pi, dim=-1)

    k = torch.multinomial(pi_temp[0, 0], num_samples=1).item()

    mu_k = mu[0, 0, k]
    sigma_k = sigma[0, 0, k]

    sample = mu_k + sigma_k * temperature * torch.randn_like(mu_k)
    return sample.unsqueeze(0).unsqueeze(0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AimGRU()

    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 17)

    output, _ = model(x)
