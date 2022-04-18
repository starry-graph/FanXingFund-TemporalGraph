import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
        '''

        super(MLP, self).__init__()

        self.num_layers = num_layers
        self.linears = nn.ModuleList()
        # We allow zero layer here, and return the input directly.
        if num_layers >= 1:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
        self.linears.extend(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)])
        self.act = nn.ReLU()

    def forward(self, x):
        if self.num_layers < 1:
            return x
        
        h = x
        for layer in range(self.num_layers - 1):
            h = self.act(self.linears[layer](h))
        h = self.linears[-1](h)
        return h