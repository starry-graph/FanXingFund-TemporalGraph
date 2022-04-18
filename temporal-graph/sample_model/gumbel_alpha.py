# Code to implement VAE-gumple_softmax in pytorch
# author: Devinder Kumar (devinder.kumar@uwaterloo.ca), modified by Yongfei Yan
# The code has been modified from pytorch example vae code and inspired by the origianl \
# tensorflow implementation of gumble-softmax by Eric Jang.

import argparse
import logging
import numpy as np
import os
import pickle

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn import functional as F
from tqdm import trange

from data_loader.data_util import _iterate_datasets
from data_loader.data_util import load_graph, load_data
from sample_model.graph import NeighborFinder
from tgat.module import (
    MergeLayer,
    ScaledDotProductAttention,
    TGAN,
)


class GumbelAttnModel(torch.nn.Module):
    """Attention based temporal layers"""
    def __init__(
        self,
        feat_dim,
        edge_dim,
        time_dim,
        n_head=1,
        drop_out=0.1,
        hard="soft",
        temp=0.1,
        num_neighbors=20,
    ):
        """
        args:
          feat_dim: dim for the node features
          edge_dim: dim for the temporal edge features
          time_dim: dim for the time encoding
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(GumbelAttnModel, self).__init__()

        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        self.hard = hard  # Use attention, soft or hard Gumbel softmax.

        self.edge_in_dim = feat_dim + edge_dim + time_dim
        self.model_dim = self.edge_in_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        # assert self.model_dim % n_head == 0
        self.logger = logging.getLogger(__name__)

        # Setup Gumbel Sampling parameters.
        self.register_buffer("num_neighbors", torch.tensor(num_neighbors))
        self.temp = temp
        self.eps = 1e-20

        # Setup the attention parameters.
        assert n_head == 1
        d_model = d_k = d_v = self.model_dim
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.attention = ScaledDotProductAttention(temperature=np.power(
            d_k, 0.5),
                                                   attn_dropout=drop_out)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(drop_out)
        self.softmax = torch.nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):
        d_model = d_k = d_v = self.model_dim
        nn.init.normal_(self.w_qs.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight,
                        mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))
        nn.init.xavier_normal_(self.fc.weight)

    def anneal_temp(self, temp):
        self.temp = temp

    def gumbel_softmax(self, logits):
        """
        ST-gumbel-softmax
        input: [batch, ngh_num]
        return: flatten --> [batch, ngh_num] an one-hot vector
        """
        temp = self.temp
        num_neighbors = self.num_neighbors

        U = torch.rand_like(logits)
        g_ = -torch.log(-torch.log(U + self.eps) + self.eps)
        y = logits + g_
        y = F.softmax(y / temp, dim=-1)

        if self.hard == "soft" or y.shape[-1] < num_neighbors:
            return y

        shape = y.size()
        _, ind = y.topk(num_neighbors, dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def attention_score(self, q, k, v, mask=None, gumbel=True):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k,
                                                    d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v,
                                                    d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        # output, attn = self.attention(q, k, v, mask=mask)
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -1e10)

        if gumbel:
            # Here, we perform gumbel sampling.
            gumbel_attn = self.gumbel_softmax(attn)
            gumbel_attn = self.dropout(gumbel_attn)
            output = torch.bmm(gumbel_attn, v)  # [n * b, l_v, d]
        else:
            attn = self.softmax(attn)  # [n * b, l_q, l_k]
            attn = self.dropout(attn)  # [n * b, l_v, d]
            output = torch.bmm(attn, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = (output.permute(1, 2, 0,
                                 3).contiguous().view(sz_b, len_q,
                                                      -1))  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        # output = output + residual
        output = self.layer_norm(output + residual)

        return output, attn

    def forward(self, src, src_t, seq, seq_t, seq_e, mask, gumbel=True):
        """Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          src_t: float Tensor of shape [B, Dt], Dt == D
          seq: float Tensor of shape [B, N, D]
          seq_t: float Tensor of shape [B, N, Dt]
          seq_e: float Tensor of shape [B, N, De], De == D
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """

        src_ext = torch.unsqueeze(src, dim=1)  # src [B, 1, D]
        # src_e_ph = torch.zeros_like(src_ext)
        src_e_ph = torch.zeros(
            (src_ext.shape[0], 1, self.edge_dim)).to(src_ext)
        q = torch.cat([src_ext, src_e_ph, src_t],
                      dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]
        k = torch.cat([seq, seq_e, seq_t],
                      dim=2)  # [B, 1, D + De + Dt] -> [B, 1, D]

        mask = torch.unsqueeze(mask, dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  # mask [B, 1, N]

        # target-attention
        # output, attn = self.multi_head_target(
        #     q=q, k=k, v=k, mask=mask
        # )  # output: [B, 1, D + Dt], attn: [B, 1, N]
        # output: [B, 1, D + Dt], attn: [B, 1, N]
        output, attn = self.attention_score(q=q,
                                            k=k,
                                            v=k,
                                            mask=mask,
                                            gumbel=gumbel)
        output = output.squeeze(1)  # When B is 1, an error occurs here.
        attn = attn.squeeze(1)

        output = self.merger(output, src)
        return output, attn


class GumbelGAN(TGAN):
    def __init__(self, *args, **kwargs):
        num_layers = kwargs.get("num_layers")
        n_head = kwargs.get("n_head")
        drop_out = kwargs.get("drop_out")
        self.hard = kwargs.pop("hard")
        self.num_neighbors = kwargs.pop("num_neighbors")
        assert (num_layers <= 1)
        assert (n_head <= 1)

        super(GumbelGAN, self).__init__(*args, **kwargs)
        self.attn_model = GumbelAttnModel(
            self.n_feat_dim,
            self.e_feat_dim,
            self.time_dim,
            n_head=n_head,
            drop_out=drop_out,
        )

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed, _ = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors)
        target_embed, _ = self.tem_conv(target_idx_l, cut_time_l,
                                        self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score

    def contrast(
        self,
        src_idx_l,
        target_idx_l,
        background_idx_l,
        cut_time_l,
        num_neighbors=20,
        gumbel=False,
    ):
        src_embed, _ = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors, gumbel)
        target_embed, _ = self.tem_conv(target_idx_l, cut_time_l,
                                        self.num_layers, num_neighbors, gumbel)
        background_embed, _ = self.tem_conv(background_idx_l, cut_time_l,
                                            self.num_layers, num_neighbors,
                                            gumbel)
        pos_score = self.affinity_score(src_embed,
                                        target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed,
                                        background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def anneal_temp(self, temp):
        self.attn_model.anneal_temp(temp)

    def gumbel_conv(self,
                    src_idx,
                    cut_time,
                    curr_layers=1,
                    num_neighbors=20,
                    gumbel=True):
        # gumbel=True for gumbel-softmax training; gumbel=False for attention
        # inference.
        assert curr_layers >= 0

        device = self.n_feat_th.device

        batch_size = len(src_idx)

        src_node_batch_th = torch.from_numpy(src_idx).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.gumbel_conv(
                src_idx,
                cut_time,
                curr_layers=curr_layers - 1,
                num_neighbors=num_neighbors,
            )

            # Only support one node per time. Train a 1-layer network to
            # sample the top-k neighbors of attention scores.
            assert (len(src_idx) == 1)
            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.find_before(
                src_idx[0], cut_time[0])
            find_neighbors = len(src_ngh_node_batch)
            if find_neighbors <= 0:
                find_neighbors = 1
                src_ngh_node_batch = np.zeros((1, ), dtype=np.int32)
                src_ngh_eidx_batch = np.zeros((1, ), dtype=np.int32)
                src_ngh_t_batch = np.zeros((1, ), dtype=np.float32)
            src_ngh_node_batch = src_ngh_node_batch[np.newaxis, :]
            src_ngh_eidx_batch = src_ngh_eidx_batch[np.newaxis, :]
            src_ngh_t_batch = src_ngh_t_batch[np.newaxis, :]

            src_ngh_node_batch_th = (
                torch.from_numpy(src_ngh_node_batch).long().to(device))
            src_ngh_eidx_batch = torch.from_numpy(
                src_ngh_eidx_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = (
                torch.from_numpy(src_ngh_t_batch_delta).float().to(device))

            # get previous layer's node features
            src_ngh_node_batch_flat = (src_ngh_node_batch.flatten()
                                       )  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten(
            )  # reshape(batch_size, -1)
            # We currently only support 1-layer because nodes have different
            # numbers of neighbors without sampling.
            src_ngh_node_conv_feat = self.gumbel_conv(
                src_ngh_node_batch_flat,
                src_ngh_t_batch_flat,
                curr_layers=curr_layers - 1,
                num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       find_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0

            local, attn = self.attn_model(
                src_node_conv_feat,
                src_node_t_embed,
                src_ngh_feat,
                src_ngh_t_embed,
                src_ngn_edge_feat,
                mask,
                gumbel=gumbel,
            )
            return local, attn

    def tem_conv(self,
                 src_idx_l,
                 cut_time_l,
                 curr_layers=1,
                 num_neighbors=20,
                 gumbel=False):
        assert curr_layers >= 0

        if gumbel:
            if self.hard == "atte":
                self.logger.warning(
                    "Args.hard denotes only attention scores, but tem_conv calls with gumbel=True."
                )
            return self.gumbel_conv(src_idx_l, cut_time_l, curr_layers,
                                    num_neighbors)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat
        else:
            src_node_conv_feat = self.tem_conv(
                src_idx_l,
                cut_time_l,
                curr_layers=curr_layers - 1,
                num_neighbors=num_neighbors,
            )

            src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
                src_idx_l, cut_time_l, num_neighbors=num_neighbors)

            src_ngh_node_batch_th = (
                torch.from_numpy(src_ngh_node_batch).long().to(device))
            src_ngh_eidx_batch = torch.from_numpy(
                src_ngh_eidx_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = (
                torch.from_numpy(src_ngh_t_batch_delta).float().to(device))

            # get previous layer's node features
            src_ngh_node_batch_flat = (src_ngh_node_batch.flatten()
                                       )  # reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten(
            )  # reshape(batch_size, -1)
            src_ngh_node_conv_feat = self.tem_conv(src_ngh_node_batch_flat,
                                                   src_ngh_t_batch_flat,
                                                   curr_layers=curr_layers - 1,
                                                   num_neighbors=num_neighbors,
                                                   gumbel=gumbel)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       num_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            # attn_m = tgan.attn_model_list[curr_layers - 1]

            # local, weight = attn_m(
            #     src_node_conv_feat,
            #     src_node_t_embed,
            #     src_ngh_feat,
            #     src_ngh_t_embed,
            #     src_ngn_edge_feat,
            #     mask,
            # )
            local, attn = self.attn_model(
                src_node_conv_feat,
                src_node_t_embed,
                src_ngh_feat,
                src_ngh_t_embed,
                src_ngn_edge_feat,
                mask,
                gumbel,
            )
            return local, attn


class GumbelNFinder(object):
    """The GumbelNFinder module uses a pretrained 1 layer TGAT module to
    compute the attention scores, then finetunes the module parameter during
    training. This module is then freezed for another GNN training.

    Reference: https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py

    TODO: Use reinforcement learning like `Pre-Training Graph Neural Networks
    for Cold-Start Users and Items Representation`.
    """

    PRECISION = 5

    def __init__(self, adj_list, gumbel_nn, hard="atte"):
        super(GumbelNFinder, self).__init__()

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(
            adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l
        self.gumbel_nn = gumbel_nn
        self.hard = hard
        self.cache = {}

        self.logger = logging.getLogger(__name__)

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]

        """
        n_idx_l = []
        n_ts_l = []
        e_idx_l = []
        off_set_l = [0]
        for i in range(len(adj_list)):
            curr = adj_list[i]
            curr = sorted(curr, key=lambda x: x[1])
            n_idx_l.extend([x[0] for x in curr])
            e_idx_l.extend([x[1] for x in curr])
            n_ts_l.extend([x[2] for x in curr])

            off_set_l.append(len(n_idx_l))
        n_idx_l = np.array(n_idx_l)
        n_ts_l = np.array(n_ts_l)
        e_idx_l = np.array(e_idx_l)
        off_set_l = np.array(off_set_l)

        assert len(n_idx_l) == len(n_ts_l)
        assert off_set_l[-1] == len(n_ts_l)

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            return neighbors_idx, neighbors_e_idx, neighbors_ts

        right = np.searchsorted(neighbors_ts, cut_time, side="left")
        ngh_idx = neighbors_idx[:right]
        ngh_eidx = neighbors_e_idx[:right]
        ngh_ts = neighbors_ts[:right]
        return ngh_idx, ngh_eidx, ngh_ts

    def gumbel_sample(self, src_idx, cut_time, num_neighbors=20):
        # attn: [batch_size, num_neighbors]
        src_idx = src_idx[np.newaxis]
        cut_time = cut_time[np.newaxis]
        _, attn = self.gumbel_nn.gumbel_conv(src_idx,
                                             cut_time,
                                             1,
                                             num_neighbors,
                                             gumbel=False)
        _, ind = attn.topk(num_neighbors, dim=-1)
        return ind.detach().cpu().numpy()

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert len(src_idx_l) == len(cut_time_l)

        out_ngh_node_batch = np.zeros(
            (len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros(
            (len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros(
            (len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            ngh_idx, ngh_eidx, ngh_ts = self.find_before(src_idx, cut_time)

            if len(ngh_idx) <= 0:
                continue
            if len(ngh_idx) < num_neighbors:
                right = len(ngh_idx)
                out_ngh_node_batch[i, :right] = ngh_idx
                out_ngh_t_batch[i, :right] = ngh_ts
                out_ngh_eidx_batch[i, :right] = ngh_eidx
                continue

            if not hasattr(self, "ngh_cache"):
                raise NotImplementedError("Use self.precompute first.")
            right = len(ngh_idx)
            out_ngh_node_batch[
                i, :] = self.ngh_cache["node_cache"][src_idx][right]
            out_ngh_t_batch[i, :] = self.ngh_cache["t_cache"][src_idx][right]
            out_ngh_eidx_batch[
                i, :] = self.ngh_cache["eidx_cache"][src_idx][right]

        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def precompute(self, task, data, num_neighbors=20, freeze=False):
        # Precompute (self.edge_idx_l, num_neighbors) for each interaction
        SAVE_PATH = f"gumbel_cache/{task}-{freeze}-{data}-gumbel-{self.hard}-{num_neighbors}.pyc"
        if os.path.exists(SAVE_PATH):
            self.logger.info("File %s exists in gumbel_cache.", SAVE_PATH)
            self.ngh_cache = pickle.load(open(SAVE_PATH, 'rb'))
            return

        self.logger.info("File %s not found in gumbel_cache.", SAVE_PATH)
        # ngh_idx_cache = [0 for _ in range(len(self.off_set_l))]
        node_cache = [0 for _ in range(len(self.off_set_l))]
        t_cache = [0 for _ in range(len(self.off_set_l))]
        eidx_cache = [0 for _ in range(len(self.off_set_l))]

        for i in trange(len(self.off_set_l) - 1):
            start = self.off_set_l[i]
            end = self.off_set_l[i + 1]

            slots = end - start + 1
            node_batch = np.zeros((slots, num_neighbors), dtype=np.int32)
            t_batch = np.zeros((slots, num_neighbors), dtype=np.float32)
            eidx_batch = np.zeros((slots, num_neighbors), dtype=np.int32)

            ngh_idx = self.node_idx_l[start:end]
            ngh_ts = self.node_ts_l[start:end]
            ngh_eidx = self.edge_idx_l[start:end]
            # Here right refers to np.searchsorted(node, timestamp).
            # For a node with n interactions, it has n+1 right slots.
            # For each (node, right), store the sampled_idx given by gumbel_sample.
            for k in range(slots):
                if slots <= 1:  # for the padding node
                    continue

                src_idx = np.array(i)
                if k < len(ngh_ts):
                    cut_time = ngh_ts[k]
                else:
                    cut_time = ngh_ts[-1] + 1  # the last timestamp

                right = np.sum(ngh_ts < cut_time)
                if right < num_neighbors:
                    # print(right)
                    node_batch[k, :right] = ngh_idx[:right]
                    t_batch[k, :right] = ngh_ts[:right]
                    eidx_batch[k, :right] = ngh_eidx[:right]
                else:
                    sampled_idx = self.gumbel_sample(src_idx, cut_time,
                                                     num_neighbors)
                    node_batch[k, :] = ngh_idx[sampled_idx]
                    t_batch[k, :] = ngh_ts[sampled_idx]
                    eidx_batch[k, :] = ngh_eidx[sampled_idx]

                    pos = t_batch[k, :].argsort()
                    node_batch[k, :] = node_batch[k, :][pos]
                    t_batch[k, :] = t_batch[k, :][pos]
                    eidx_batch[k, :] = eidx_batch[k, :][pos]

            # Completion
            node_cache[i] = node_batch
            t_cache[i] = t_batch
            eidx_cache[i] = eidx_batch

        ngh_cache = {
            "node_cache": node_cache,
            "t_cache": t_cache,
            "eidx_cache": eidx_cache
        }
        # np.savez(SAVE_PATH, ngh_cache, allow_pickle=True)
        pickle.dump(ngh_cache, open(SAVE_PATH, 'wb'))
        self.ngh_cache = ngh_cache
