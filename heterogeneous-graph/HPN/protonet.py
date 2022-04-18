import torch.nn as nn
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 4),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.MaxPool1d(2)
    )


class Heteproto(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(Heteproto, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, features):
        for k in features.keys():
            x = torch.unsqueeze(features[k],1)
            print(x.size())
            features[k] = self.encoder(x)
            features[k] = features[k].view(features[k].size(0), -1)
            print(features[k].size())
        
        return features
