import torch
import torch.nn as nn


class AimGRU(nn.Module):
    def __init__(
        self,
        input_size: int = 8,
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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AimGRU()
    print(f"Model created: {count_parameters(model):,} parameters")

    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, 8)

    output, _ = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, seq_len, 2), f"Expected (4, 512, 2), got {output.shape}"
    print("All tests passed!")
