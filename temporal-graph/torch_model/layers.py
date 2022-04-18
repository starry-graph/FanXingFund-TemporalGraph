import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import dgl.function as fn

class TimeEncode(torch.nn.Module):
    def __init__(self, time_dim):
        super(TimeEncode, self).__init__()
        self.basis_freq = nn.Parameter(
                torch.linspace(0, 9, time_dim))
        self.phase = nn.Parameter(torch.zeros(time_dim).float())

    def forward(self, ts):
        ts = ts.view(-1, 1)
        map_ts = ts * self.basis_freq.view(1, -1)
        map_ts += self.phase.view(1, -1)
        harmonic = torch.cos(map_ts)
        return harmonic


class TimeEncodingLayer(nn.Module):
    """Given `(E_u, t)`, output `f2(act(f1(E_u, Encode(t))))`.
    """

    def __init__(self, in_features, out_features, time_encoding="concat"):
        super(TimeEncodingLayer, self).__init__()
        self.time_encoding = time_encoding
        if time_encoding == "concat":
            self.fc1 = nn.Linear(in_features + 1, out_features)
        elif time_encoding == "empty":
            self.fc1 = nn.Linear(in_features, out_features)
        elif time_encoding == "cosine":
            self.basis_freq = nn.Parameter(
                torch.linspace(0, 9, out_features))
            self.phase = nn.Parameter(torch.zeros(out_features))
            self.fc1 = nn.Linear(in_features + out_features, out_features)
        else:
            raise NotImplementedError
        self.act = nn.ReLU()

        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, u, t):
        if self.time_encoding == "concat":
            x = self.fc1(torch.cat([u, t.view(-1, 1)], dim=1))
        elif self.time_encoding == "empty":
            x = self.fc1(u)
        elif self.time_encoding == "cosine":
            t = torch.cos(t.view(-1, 1) * self.basis_freq.view(1, -1) +
                          self.phase.view(1, -1))
            x = self.fc1(torch.cat([u, t], dim=-1))
        else:
            raise NotImplementedError

        return self.act(x)


class TemporalLinkLayer(nn.Module):
    """Given a list of `(u, v, t)` tuples, predicting the edge probability between `u` and `v` at time `t`. Firstly, we find the latest `E(u, t_u)` and `E(v, t_v)` before the time `t`. Then we compute `E(u, t)` and `E(v, t)` using an outer product temporal encoding layer for `E(u, t_u)` and `E(v, t_v)` respectively. Finally, we concatenate the embeddings and output probability logits via a two layer MLP like `TGAT`.
    """

    def __init__(self, in_features=128, out_features=1, concat=True, time_encoding="concat", dropout=0.2, proj=True):
        super(TemporalLinkLayer, self).__init__()
        self.concat = concat
        self.time_encoding = time_encoding
        mul = 2 if concat else 1
        self.time_encoder = TimeEncodingLayer(
            in_features, in_features, time_encoding=time_encoding)
        self.fc = nn.Linear(in_features * mul, out_features)
        self.dropout = nn.Dropout(dropout)
        self.proj = proj

    def forward(self, g, src_eids, dst_eids, t):
        """For each `(u, v, t)`, we get embedding_u by
        `g.edata['src_feat'][src_eids]`, get embedding_v by
        `g.edata['dst_feat'][dst_eids]`.

        Finally, output `g(e_u, e_v, t)`.
        """
        featu = g.edata["src_feat"][src_eids]
        tu = g.edata["timestamp"][src_eids]
        featv = g.edata["dst_feat"][dst_eids]
        tv = g.edata["timestamp"][dst_eids]
        if self.proj:
            embed_u = self.time_encoder(featu, t-tu)
            embed_v = self.time_encoder(featv, t-tv)
        else:
            embed_u, embed_v = featu, featv

        if self.concat:
            x = torch.cat([embed_u, embed_v], dim=1)
        else:
            x = embed_u + embed_v
        logits = self.fc(self.dropout(x))
        return logits.squeeze()

class FastTSAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, aggregator_type):
        super(FastTSAGEConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        # self.encode_time = TimeEncodingLayer(
        #     in_feats, in_feats, time_encoding=time_encoding)
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats,
                                self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, edge_feat):
        """LSTM processing for temporal edges.
        """
        batch_size = edge_feat.shape[0]
        h = (edge_feat.new_zeros((1, batch_size, self._in_src_feats)),
             edge_feat.new_zeros((1, batch_size, self._in_src_feats)))
        rst, (h_, c_) = self.lstm(edge_feat, h)
        return rst

    def group_func_wrapper(self, src_feat):
        """Instead, we always perfrom src->dst convolution. Also, the transformation works are left for ``forward`` function for speedup."""
        def onehop_conv(edges):
            h_neighs = edges.data[src_feat]

            if self._aggre_type == "mean":
                h_feat = h_neighs.cumsum(dim=1)
            elif self._aggre_type == "gcn":
                h_feat = h_neighs.cumsum(dim=1)
            elif self._aggre_type == "pool":
                # Transformation is retrieved.
                h_feat = h_neighs.cummax(dim=1).values
            elif self._aggre_type == 'lstm':
                h_feat = self._lstm_reducer(h_neighs)
            else:
                raise NotImplementedError
            
            return {"h_neigh": h_feat}
        return onehop_conv

    def forward(self, graph, current_layer):
        """For each edge (src, dst, t), obtain the convolution results CONV(``(src', dst, t_i) and t_i \le t``)."""
        g = graph.local_var()

        # src_feat is composed of [node_feat, edge_feat, time_encoding].
        src_name = f'src_feat{current_layer - 1}'
        dst_name = f'dst_feat{current_layer - 1}'
        src_feat = g.edata[src_name]

        if self._aggre_type == "pool":
            # Transform before batching.
            g.edata[src_name] = F.relu(self.fc_pool(src_feat))

        dst_conv = self.group_func_wrapper(src_feat=src_name)
        g.group_apply_edges(group_by="dst", func=dst_conv)
        # Each edge accumulates the historical embeddings. While there exist edges with the same time point. Therefore, we fetch the correct h_neigh here.
        h_neigh = g.edata["h_neigh"][g.edata["dst_max_eid"]]
        h_self = g.edata[dst_name]

        if self._aggre_type == "mean":
            mean_cof = g.edata["dst_deg"].add(1.0)
            h_neigh = h_neigh / mean_cof.unsqueeze(-1)
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        elif self._aggre_type == "gcn":
            norm_cof = g.edata["dst_deg"].add(1.0)
            h_neigh = (h_neigh + h_self) / norm_cof.unsqueeze(-1)
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

        return rst
