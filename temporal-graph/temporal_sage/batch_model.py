"""RGCN layer implementation"""
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as fn
import tqdm
import torch

from model import DotProductPredictor


class BatchSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation,
                 dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation,
             dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class BatchModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers):
        super().__init__()
        self.sage = BatchSAGE(in_features, hidden_features, out_features, n_layers, nn.ReLU(), 0.2)
        self.lstm = nn.LSTM(out_features, out_features, 1)
        self.pred = DotProductPredictor()
        self.out_features = out_features

    def forward(self, history_blocks, history_inputs, pos_g, neg_g):
        hs = [self.sage(g, x) for g, x in zip(history_blocks, history_inputs)]
        hs = torch.stack(hs) # (seq, batch, dim)
        hs, _ = self.lstm(hs) # default (h0, c0) are all zeros
        h = hs[-1]
        return nn.Sigmoid()(self.pred(pos_g, h)), nn.Sigmoid()(self.pred(neg_g, h))
