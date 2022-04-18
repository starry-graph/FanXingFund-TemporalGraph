import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tgat.module import AttnModel, LSTMPool, MeanPool, TGAN


# Reference: KDD 2020 AM-GCN: Adaptive Multi-channel Graph Convolutional Networks
class SoftmaxAttention(nn.Module):
    def __init__(self, feat_dim: int, samplers: int) -> None:
        super(SoftmaxAttention, self).__init__()
        self.trans = nn.ModuleList(
            [nn.Linear(feat_dim, feat_dim) for _ in range(samplers)])
        self.k_samplers = samplers
        self.query = nn.Linear(feat_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, embeds: list) -> torch.Tensor:
        k = len(embeds)
        assert len(embeds[0].shape) == 2
        x = [torch.tanh(self.trans[i](embeds[i])) for i in range(k)]
        x = [self.query(x[i]) for i in range(k)]  # (k, n, 1)
        weights = torch.softmax(torch.cat(x, dim=1), dim=1)  # (n, k)
        embeds = torch.cat([i.unsqueeze(dim=1) for i in embeds],
                           dim=1)  # (n, k, d)
        ans = weights.unsqueeze(-1) * embeds  # (n, k, d)
        ans = self.layer_norm(ans.sum(dim=1))
        return ans, weights  # (n, d)


class SamplingFusion(TGAN):
    def __init__(self, *args, **kwargs) -> None:
        # [Time Decay Sampling, Gumbel Attention Sampling]
        self.k_samplers = kwargs.pop("k_samplers")
        self.num_layers = kwargs['num_layers']

        # For each layer, we employ k attention models and a fusion layer.
        super(SamplingFusion, self).__init__(*args, **kwargs)
        delattr(self, "attn_model_list")
        agg_method = kwargs["agg_method"]
        attn_mode = kwargs["attn_mode"]
        n_head = kwargs["n_head"]
        drop_out = kwargs["drop_out"]
        self.attn_model_list = nn.ModuleList([
            self.create_attn_model(agg_method, attn_mode, n_head, drop_out)
            for _ in range(self.k_samplers)
        ])

        feat_dim = self.feat_dim
        self.fusion_layer_list = torch.nn.ModuleList([
            SoftmaxAttention(feat_dim, self.k_samplers)
            for _ in range(self.num_layers)
        ])

    def create_attn_model(self, agg_method, attn_mode, n_head, drop_out):
        # For each layer, we employ k attention models and a fusion layer.
        n_feat_dim = self.n_feat_dim
        e_feat_dim = self.e_feat_dim
        time_dim = self.time_dim
        num_layers = self.num_layers

        if agg_method == 'attn':
            # self.logger.info('Aggregation uses attention model')
            attn_model_list = nn.ModuleList([
                AttnModel(n_feat_dim,
                          e_feat_dim,
                          time_dim,
                          attn_mode=attn_mode,
                          n_head=n_head,
                          drop_out=drop_out) for _ in range(num_layers)
            ])
        elif agg_method == 'lstm':
            # self.logger.info('Aggregation uses LSTM model')
            attn_model_list = nn.ModuleList([
                LSTMPool(n_feat_dim, e_feat_dim, time_dim)
                for _ in range(num_layers)
            ])
        elif agg_method == 'mean':
            # self.logger.info('Aggregation uses constant mean model')
            attn_model_list = nn.ModuleList(
                [MeanPool(n_feat_dim, e_feat_dim) for _ in range(num_layers)])
        else:

            raise ValueError('invalid agg_method value, use attn or lstm')
        return attn_model_list

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):
        src_embed, _ = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors)
        target_embed, _ = self.tem_conv(target_idx_l, cut_time_l,
                                        self.num_layers, num_neighbors)
        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        return score

        # src_l_cut, dst_l_cut, dst_l_fake,ts_l_cut, NUM_NEIGHBORS
    def contrast(self, src_idx_l, target_idx_l, background_idx_l, cut_time_l,
                 num_neighbors):
        src_embed, _ = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors)
        target_embed, _ = self.tem_conv(target_idx_l, cut_time_l,
                                        self.num_layers, num_neighbors)
        background_embed, _ = self.tem_conv(background_idx_l, cut_time_l,
                                            self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed,
                                        target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed,
                                        background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self,
                 src_idx_l,
                 cut_time_l,
                 curr_layers,
                 num_neighbors=20) -> torch.Tensor:
        """Here we precomputed the k-hop neighbors instead of computing during attention models.
        """
        assert (curr_layers >= 0)
        assert num_neighbors % self.k_samplers == 0
        device = self.n_feat_th.device
        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)
        cut_time_l_th = torch.unsqueeze(cut_time_l_th, dim=1)
        # query node always has the start time -> time span == 0
        src_node_t_embed = self.time_encoder(torch.zeros_like(cut_time_l_th))
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat, torch.full((batch_size, 2), 0.5).to(device)

        # get node features at previous layer
        src_node_conv_feat, _ = self.tem_conv(src_idx_l,
                                              cut_time_l,
                                              curr_layers=curr_layers - 1,
                                              num_neighbors=num_neighbors)
        ngh_batch = self.ngh_finder.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors)
        # next layer also perform sampling fusion
        sampling_feats = []
        for k, (src_ngh_node_batch, src_ngh_eidx_batch,
                src_ngh_t_batch) in enumerate(ngh_batch):
            # Specified attention model for the k-th sampler
            half_neighbors = num_neighbors // self.k_samplers

            attn_model_k = self.attn_model_list[k]
            attn_m = attn_model_k[curr_layers - 1]

            src_ngh_node_batch_th = torch.from_numpy(
                src_ngh_node_batch).long().to(device)
            src_ngh_eidx_batch = torch.from_numpy(
                src_ngh_eidx_batch).long().to(device)

            src_ngh_t_batch_delta = cut_time_l[:, np.newaxis] - src_ngh_t_batch
            src_ngh_t_batch_th = torch.from_numpy(
                src_ngh_t_batch_delta).float().to(device)

            # get previous layer's node features
            src_ngh_node_batch_flat = src_ngh_node_batch.flatten(
            )  #reshape(batch_size, -1)
            src_ngh_t_batch_flat = src_ngh_t_batch.flatten(
            )  #reshape(batch_size, -1)
            src_ngh_node_conv_feat, _ = self.tem_conv(
                src_ngh_node_batch_flat,
                src_ngh_t_batch_flat,
                curr_layers=curr_layers - 1,
                num_neighbors=num_neighbors)
            src_ngh_feat = src_ngh_node_conv_feat.view(batch_size,
                                                       half_neighbors, -1)

            # get edge time features and node features
            src_ngh_t_embed = self.time_encoder(src_ngh_t_batch_th)
            src_ngn_edge_feat = self.edge_raw_embed(src_ngh_eidx_batch)

            # attention aggregation
            mask = src_ngh_node_batch_th == 0
            local, weight = attn_m(src_node_conv_feat, src_node_t_embed,
                                   src_ngh_feat, src_ngh_t_embed,
                                   src_ngn_edge_feat, mask)
            sampling_feats.append(local)

        # fuse feats under different sampling strategies
        fusion_layer = self.fusion_layer_list[curr_layers - 1]
        fusion_feats, score = fusion_layer(sampling_feats)
        return fusion_feats, score
