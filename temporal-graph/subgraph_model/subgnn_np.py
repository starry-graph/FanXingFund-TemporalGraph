import logging

import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

from subgraph_model.graph import batch_interaction2subgraph
from subgraph_model.mlp import MLP
from tgat.module import TimeEncode, MapBasedMultiHeadAttention, MultiHeadAttention, TGAN, MergeLayer


class SimpleAttention(torch.nn.Module):
    """Variant of attention based temporal layers
    """
    def __init__(self, feat_dim, attn_mode='prod', n_head=2, drop_out=0.1):
        """
        args:
          feat_dim: dim for the node features
          attn_mode: choose from 'prod' and 'map'
          n_head: number of heads in attention
          drop_out: probability of dropping a neural.
        """
        super(SimpleAttention, self).__init__()

        self.feat_dim = feat_dim
        self.model_dim = feat_dim

        self.merger = MergeLayer(self.model_dim, feat_dim, feat_dim, feat_dim)

        assert (self.model_dim % n_head == 0)
        self.logger = logging.getLogger(__name__)
        self.attn_mode = attn_mode

        if attn_mode == 'prod':
            self.multi_head_target = MultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out)
        elif attn_mode == 'map':
            self.multi_head_target = MapBasedMultiHeadAttention(
                n_head,
                d_model=self.model_dim,
                d_k=self.model_dim // n_head,
                d_v=self.model_dim // n_head,
                dropout=drop_out)
        else:
            raise ValueError('attn_mode can only be prod or map')

    def forward(self, src, seq, mask):
        """"Attention based temporal attention forward pass
        args:
          src: float Tensor of shape [B, D]
          seq: float Tensor of shape [B, N, D]
          mask: boolean Tensor of shape [B, N], where the true value indicate a null value in the sequence.

        returns:
          output, weight

          output: float Tensor of shape [B, D]
          weight: float Tensor of shape [B, N]
        """
        q = src.unsqueeze(dim=1)  # src [B, 1, D]
        k = seq  # neighbor [B, N, D]

        mask = mask.unsqueeze(dim=2)  # mask [B, N, 1]
        mask = mask.permute([0, 2, 1])  #mask [B, 1, N]

        # target-attention
        output, attn = self.multi_head_target(
            q=q, k=k, v=k, mask=mask)  # output: [B, 1, D], attn: [B, 1, N]
        output = output.squeeze(1)  # When B is 1, an error occurs here.
        attn = attn.squeeze(1)

        output = self.merger(output, src)
        return output, attn


# Reference: KDD 2020 AM-GCN: Adaptive Multi-channel Graph Convolutional Networks
class SoftmaxAttention(nn.Module):
    def __init__(self, feat_dim: int, num: int) -> None:
        super(SoftmaxAttention, self).__init__()
        self.trans = nn.Linear(feat_dim * num, feat_dim * num, bias=False)
        self.num = num
        self.query = nn.Linear(feat_dim, 1, bias=False)
        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(self, embeds: list) -> torch.Tensor:
        # embeds: (num, (batch, dim))
        num = len(embeds)
        batch, dim = embeds[0].shape
        # x = torch.cat([e.unsqueeze(dim=1) for e in embeds], dim=1)
        x = torch.stack(embeds, dim=1)
        trans_x = self.trans(x.view(batch, num * dim)).tanh()
        weights = self.query(trans_x.view(batch, num, dim))
        weights = torch.softmax(weights.view(batch, num), dim=1)
        ans = torch.bmm(weights.unsqueeze(1),
                        x)  # (batch, 1, num) * (batch, num, dim)
        ans = self.layer_norm(ans.sum(dim=1))
        return ans, weights


class SubgraphConv(nn.Module):
    def __init__(self,
                 nfeat_dim,
                 efeat_dim,
                 num_prop=3,
                 num_mlp_layers=2,
                 alpha=0.2) -> None:
        super(SubgraphConv, self).__init__()
        self.num_layer = num_prop + 1
        self.nfeat_dim = nfeat_dim
        self.model_dim = nfeat_dim
        self.efeat_dim = efeat_dim
        # self.edge_fc = nn.Linear(efeat_dim, nfeat_dim)
        self.edge_merger = MergeLayer(nfeat_dim, efeat_dim, nfeat_dim, nfeat_dim)
        self.mlps = nn.ModuleList([
            MLP(num_mlp_layers, nfeat_dim, nfeat_dim) for _ in range(num_prop)
        ])
        self.alpha = alpha
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(nfeat_dim) for _ in range(self.num_layer)])
        self.act = nn.ReLU()

    def forward(self, n2n, nfeat, e2n, efeat):
        batch_size, num_neighbors, nfeat_dim = nfeat.shape
        # efeat = self.edge_fc(efeat)  # (B, K, D)
        flat_nfeat = nfeat.view(batch_size * num_neighbors, -1)
        node_efeat = torch.bmm(e2n, efeat)
        flat_efeat = node_efeat.view(batch_size * num_neighbors, -1)
        h = self.edge_merger(flat_nfeat, flat_efeat)
        h = h.view(batch_size, num_neighbors, nfeat_dim)
        # h = nfeat + torch.bmm(e2n, efeat) # (B, K, D)
        feats = [h]

        degs = (n2n > 0).sum(dim=2)
        for i in range(self.num_layer - 1):
            h_next = torch.bmm(n2n, h)  # (B, K, D)
            h_next = self.mlps[i](h_next)
            h_next = self.act(self.layer_norms[i](h_next))
            h = self.alpha * h + (1 - self.alpha) * h_next
            feats.append(h)
        return feats


class SubGnnNp(nn.Module):
    def __init__(self,
                 ngh_finder,
                 n_feat,
                 e_feat,
                 n_feat_freeze=True,
                 attn_mode='prod',
                 num_layers=1,
                 num_prop=3,
                 num_mlp_layers=2,
                 alpha=0.2,
                 n_head=1,
                 null_idx=0,
                 drop_out=0.1):
        super(SubGnnNp, self).__init__()

        self.num_layers = num_layers
        self.ngh_finder = ngh_finder
        self.null_idx = null_idx
        self.logger = logging.getLogger(__name__)

        self.n_feat_th = torch.nn.Parameter(
            torch.from_numpy(n_feat.astype(np.float32)))
        self.e_feat_th = torch.nn.Parameter(
            torch.from_numpy(e_feat.astype(np.float32)))
        self.edge_raw_embed = torch.nn.Embedding.from_pretrained(
            self.e_feat_th, padding_idx=0, freeze=True)
        self.node_raw_embed = torch.nn.Embedding.from_pretrained(
            self.n_feat_th, padding_idx=0, freeze=n_feat_freeze)

        # We set n_feat, time_dim, and hidden dimension as the same.
        self.feat_dim = self.n_feat_th.shape[1]
        self.n_feat_dim = self.n_feat_th.shape[1]
        self.e_feat_dim = self.e_feat_th.shape[1]
        self.time_dim = self.feat_dim
        self.model_dim = self.feat_dim

        self.merge_layer = MergeLayer(self.feat_dim, self.feat_dim,
                                      self.feat_dim, self.feat_dim)

        self.logger.info('Aggregation uses attention model')
        self.num_layer = num_layers
        self.num_prop_layer = num_prop + 1  # includes the 0th layer
        self.num_mlp_layers = num_mlp_layers
        self.alpha = alpha
        edge_in_dim = self.e_feat_dim + self.time_dim
        self.graph_conv_list = nn.ModuleList([
            SubgraphConv(self.feat_dim, edge_in_dim, num_prop, num_mlp_layers,
                         alpha) for _ in range(num_layers)
        ])
        self.n_head = n_head
        self.attn_model_list = nn.ModuleList()
        # For each layer, we perform attention for each subgraph propagation.
        if attn_mode == "prod":
            self.logger.info('Using scaled prod attention')
        elif attn_mode == "map":
            self.logger.info('Using map based attention')
        else:
            raise ValueError('attn_mode can only be prod or map')

        for _ in range(num_layers):
            attn_layer = nn.ModuleList([
                SimpleAttention(self.n_feat_dim,
                                attn_mode=attn_mode,
                                n_head=n_head,
                                drop_out=drop_out)
                for _ in range(self.num_prop_layer)
            ])
            self.attn_model_list.append(attn_layer)
        self.fusion_layer = nn.ModuleList([
            SoftmaxAttention(self.feat_dim, self.num_prop_layer)
            for _ in range(num_layers)
        ])

        self.time_encoder = TimeEncode(expand_dim=self.time_dim)
        self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim,
                                         self.feat_dim, 1)

    def forward(self, src_idx_l, target_idx_l, cut_time_l, num_neighbors=20):

        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                  num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors)

        score = self.affinity_score(src_embed, target_embed).squeeze(dim=-1)

        return score.sigmoid()

    def contrast(self,
                 src_idx_l,
                 target_idx_l,
                 background_idx_l,
                 cut_time_l,
                 num_neighbors=20):
        src_embed = self.tem_conv(src_idx_l, cut_time_l, self.num_layers,
                                  num_neighbors)
        target_embed = self.tem_conv(target_idx_l, cut_time_l, self.num_layers,
                                     num_neighbors)
        background_embed = self.tem_conv(background_idx_l, cut_time_l,
                                         self.num_layers, num_neighbors)
        pos_score = self.affinity_score(src_embed,
                                        target_embed).squeeze(dim=-1)
        neg_score = self.affinity_score(src_embed,
                                        background_embed).squeeze(dim=-1)
        return pos_score.sigmoid(), neg_score.sigmoid()

    def tem_conv(self, src_idx_l, cut_time_l, curr_layers, num_neighbors=20):
        assert (curr_layers >= 0)

        device = self.n_feat_th.device

        batch_size = len(src_idx_l)

        src_node_batch_th = torch.from_numpy(src_idx_l).long().to(device)
        cut_time_l_th = torch.from_numpy(cut_time_l).float().to(device)

        cut_time_l_th = cut_time_l_th.unsqueeze(dim=1)
        src_node_feat = self.node_raw_embed(src_node_batch_th)

        if curr_layers == 0:
            return src_node_feat

        src_node_conv_feat = self.tem_conv(src_idx_l,
                                           cut_time_l,
                                           curr_layers=curr_layers - 1,
                                           num_neighbors=num_neighbors)

        # For simplicity, we set K=M here.
        # batch_n2n: (B, K, K)
        # batch_nid: (B, K)
        # batch_e2n: (B, K, M)
        # batch_eid: (B, M)
        # batch_ets: (B, M)
        # batch_n2n, batch_nid, batch_e2n, batch_eid, batch_ets = self.ngh_finder.get_neighbor_np(
        #     src_idx_l, cut_time_l, num_neighbors=num_neighbors)
        # src_ngh_node_batch, src_ngh_eidx_batch, src_ngh_t_batch = self.ngh_finder.get_temporal_neighbor(
        #     src_idx_l, cut_time_l, num_neighbors=num_neighbors)
        # batch_n2n, batch_nid, batch_e2n, batch_eid = batch_interaction2subgraph(
        #     src_ngh_node_batch, src_ngh_eidx_batch)
        src_ngh_t_batch, batch_subgraph = self.ngh_finder.batch_interaction2subgraph(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors)
        batch_n2n, batch_nid, batch_e2n, batch_eid = batch_subgraph
        batch_ets = src_ngh_t_batch

        flat_ngh_nid = batch_nid.flatten()
        flat_ngh_t = np.repeat(cut_time_l[:, np.newaxis],
                               num_neighbors,
                               axis=1).flatten()

        src_ngh_conv_feat = self.tem_conv(flat_ngh_nid,
                                          flat_ngh_t,
                                          curr_layers=curr_layers - 1,
                                          num_neighbors=num_neighbors)
        src_ngh_conv_feat = src_ngh_conv_feat.view(batch_size, num_neighbors,
                                                   -1)

        th_ngh_nid = torch.from_numpy(batch_nid).long().to(device)
        th_ngh_eid = torch.from_numpy(batch_eid).long().to(device)
        th_ngh_t_delta = cut_time_l[:, np.newaxis] - batch_ets
        # assert np.all(th_ngh_t_delta >= 0)
        th_ngh_t = torch.from_numpy(th_ngh_t_delta).float().to(device)
        # Here th_ngh_nid, and th_ngh_eid are not aligned, and are connected
        # by an incidence matrix: th_mat_e2n instead. We thus perfrom subgraph
        # convolution and pooling here.
        th_mat_n2n = torch.from_numpy(batch_n2n).float().to(device)
        th_mat_e2n = torch.from_numpy(batch_e2n).float().to(device)
        src_ngh_t_embed = self.time_encoder(th_ngh_t)
        src_ngh_edge_feat = self.edge_raw_embed(th_ngh_eid)
        src_ngh_efeat = torch.cat([src_ngh_t_embed, src_ngh_edge_feat],
                                  dim=-1)  # (B, K, Dt + De)

        subgraph_conv = self.graph_conv_list[curr_layers - 1]
        ngh_feats = subgraph_conv(th_mat_n2n, src_ngh_conv_feat, th_mat_e2n,
                                  src_ngh_efeat)
        curr_attn_m = self.attn_model_list[curr_layers - 1]
        # attention aggregation
        src_feats = []
        mask = th_ngh_nid == 0
        for attn_m, ngh_feat in zip(curr_attn_m, ngh_feats):
            local, weight = attn_m(src_node_conv_feat, ngh_feat, mask)
            src_feats.append(local)

        fusion = self.fusion_layer[curr_layers - 1]
        ans, weights = fusion(src_feats)  # (B, D)
        return ans
