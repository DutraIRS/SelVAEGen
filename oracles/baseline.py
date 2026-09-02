import torch
import torch.nn as nn


class BaselineOracle(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden, output_dim):
        super(BaselineOracle, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_hidden - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        mol_fps = data.fingerprint
        prot_embs = data.prot_emb
        x = torch.cat([mol_fps, prot_embs], dim=1)
        return self.mlp(x)