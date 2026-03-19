import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes,dropout_rate=0.3, batch_norm=True):
        super(MLP, self).__init__()

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.layers = nn.ModuleList()

        # Primo layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if batch_norm:
            self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.Dropout(dropout_rate))

        # Layer intermedi
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            self.layers.append(nn.Tanh())
            self.layers.append(nn.Dropout(dropout_rate))

        # Output layer — 1 neurone per classificazione binaria (logit, no sigmoid)
        self.layers.append(nn.Linear(hidden_sizes[-1], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x