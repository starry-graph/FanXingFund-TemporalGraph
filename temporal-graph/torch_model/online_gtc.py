import argparse
import logging
import os
import random
import time
from datetime import datetime

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader.data_util import load_data, load_label_edges, load_split_edges
from utils.util import EarlyStopMonitor, RandEdgeSampler, get_free_gpu, timeit
from numba import jit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import trange
from utils.util import get_free_gpu, set_logger, set_random_seed, write_result

from torch_model.dataset import TemporalDataset
from torch_model.layers import TimeEncodingLayer
from torch_model.util_dgl import construct_dglgraph

from .fast_gtc import fastgtc_args, prepare_dataset, precompute_maxeid, FastTemporalLinkTrainer

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class OnlineGConv(nn.Module):
    def __init__(self, in_feats, out_feats, agg_type) -> None:
        """
        Transform the FastTSAGEConv module parameters into OnlineGConv. We
        leave the time_encoder to OnlineSAGE, and focus on MessagePassing here.
        The messages are stored in edges, and computed as h_neigh. For
        different convolution kernels, we perform incremental computation.

        conv: h_self <- h_self + MEAN/POOL/GCN(h_neigh)
            Initialize:
                history_deg <- zeros()
                history_neigh <- zeros()
                new_deg <- graph.in_degrees()
                new_neigh <- MessagePassing()
                h_self <- nfeat
            MEAN: 
                new_neigh <- MessagePassing(graph)
                h_neigh <- SUM(new_neigh, history_neigh)
                h_neigh <- h_neigh / ADD(new_deg, history_deg)
                h_self <- fc_self(h_self) + fc_neigh(h_neigh)
            POOL: 
                new_neigh <- F.relu(fc_pool(new_neigh))
                new_neigh <- MessagePassing(graph)
                h_neigh <- MAX(new_neigh, history_neigh)
                h_self <- fc_self(h_self) + fc_neigh(h_neigh)
            GCN:  
                new_neigh <- MessagePassing(graph)
                h_neigh <- ADD(new_neigh, history_neigh)
                h_self <- (h_self + h_neigh) / ADD(new_deg, history_deg)
                h_self <- fc_neigh(h_self)
        """
        super(OnlineGConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._agg_type = agg_type
        if agg_type == 'pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if agg_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats,
                                self._in_src_feats, batch_first=True)
        if agg_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._agg_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._agg_type == 'lstm':
            self.lstm.reset_parameters()
        if self._agg_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def _lstm_reducer(self, new_neigh, h_, c_):
        new_neigh = new_neigh.unsqueeze(1) # (batch, seq, feat)
        h_ = h_.unsqueeze(0)
        c_ = c_.unsqueeze(0)
        # print(new_neigh.shape, h_.shape, c_.shape)
        rst, (h_, c_) = self.lstm(new_neigh, (h_, c_))
        return rst.squeeze(1), h_.squeeze(0), c_.squeeze(0)
    
    def _node_conv(self, graph, cur_layer=0):
        g = graph.local_var()
        src_name = f"h_self{cur_layer}"

        if self._agg_type == "pool":
            g.ndata[src_name] = F.relu(self.fc_pool(g.ndata[src_name]))
            g.update_all(fn.copy_u(src_name, "m"), fn.max("m", "new_neigh"))
        else:
            g.update_all(fn.copy_u(src_name, "m"), fn.sum("m", "new_neigh"))

        return g.ndata["new_neigh"]

    def _edge_conv(self, graph, cur_layer=0):
        g = graph.local_var()
        src_name = f"h_edge{cur_layer}"

        if self._agg_type == "pool":
            g.edata[src_name] = F.relu(self.fc_pool(g.edata[src_name]))
            g.update_all(fn.copy_e(src_name, "m"), fn.max("m", "new_neigh"))
        else:
            g.update_all(fn.copy_e(src_name, "m"), fn.sum("m", "new_neigh"))

        return g.ndata["new_neigh"]

    def forward(self, graph, cur_layer=0):
        # Now, we have history_deg and history_neigh of each layer.
        # But we don't have new_deg and new_h_neigh.
        g = graph.local_var()

        h_self = g.ndata[f"h_self{cur_layer}"]
        hist_neigh = g.ndata[f"history_neigh{cur_layer}"]
        deg = g.ndata["history_deg"] + g.in_degrees().to(h_self)
        deg = deg.add(1.0)

        # For the 0th layer, we combine node_feat, edge_feat and t_encoding
        # into the same feature in edges, named h_edge0.
        if cur_layer == 0:
            new_neigh = self._edge_conv(graph, cur_layer)
        else:
            new_neigh = self._node_conv(graph, cur_layer)

        if self._agg_type == "pool":
            hist_neigh = torch.max(hist_neigh, new_neigh)
        else:
            hist_neigh = hist_neigh + new_neigh

        if self._agg_type == "mean":
            h_neigh = hist_neigh / deg.unsqueeze(-1)
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        elif self._agg_type == "gcn":
            h_neigh = (hist_neigh + h_self) / deg.unsqueeze(-1)
            rst = self.fc_neigh(h_neigh)
        elif self._agg_type == "lstm":
            # The hidden states are stored in g.ndata.
            h_ = g.ndata[f"lstm_h{cur_layer}"]
            c_ = g.ndata[f"lstm_c{cur_layer}"]
            rst, h_, c_ = self._lstm_reducer(new_neigh, h_, c_)
        elif self._agg_type == "pool":
            rst = self.fc_self(h_self) + self.fc_neigh(hist_neigh)

        # Record current unnormolized hist_neigh for next computation.
        if self._agg_type == "lstm":
            return rst, h_, c_
        else:
            return rst, hist_neigh

class OnlineSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 edge_feats,
                 n_layers,
                 activation,
                 dropout,
                 agg_type="mean",
                 time_encoding="cosine") -> None:
        super(OnlineSAGE, self).__init__()
        self.layers = nn.ModuleList()

        self.time_encoder = TimeEncodingLayer(in_feats + edge_feats, n_hidden,
                                            time_encoding=time_encoding)
        for i in range(n_layers):
            self.layers.append(OnlineGConv(n_hidden, n_hidden, agg_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph):
        # We modify the graph data inplace here for incremental computation.
        # Init: nfeat, last_time, efeat, timestamp
        # State: history_deg, h_edge0, history_neigh{0|1|2}, h_self{0|1|2}
        # LSTM: lstm_h{0|1|2}, lstm_c{0|1|2}
        g = graph.local_var()
        # history_neigh has been set.
        g.update_all(fn.copy_e("timestamp", "m"), fn.max("m", "cur_time"))
        g.ndata["last_time"] = torch.max(g.ndata["last_time"], g.ndata["cur_time"])

        def combine_feats(edges):
            return {
                "h_edge":
                torch.cat([edges.src["nfeat"], edges.data["efeat"]], dim=1)
            }
        g.apply_edges(func=combine_feats)
        h_edge = self.time_encoder(g.edata["h_edge"], g.edata["timestamp"])
        g.edata["h_edge0"] = h_edge
        # In the implementation of FastGTC, h_self0 is initialized with 
        # src_feat, which is the same as h_edge here.
        rg = g.reverse(share_edata=True)
        def simple_reduce(nodes): return {"h_self0": 
            nodes.mailbox["m"][:, -1, :].squeeze(1)}
        rg.update_all(fn.copy_e("h_edge0", "m"), simple_reduce)
        g.ndata["h_self0"] = rg.ndata["h_self0"]
        

        # Remember to detach the tensor after loss.backward().
        for i, layer in enumerate(self.layers):
            cl = i
            if layer._agg_type == "lstm":
                h_self, h_, c_ = layer(g, cur_layer=cl)
                g.ndata[f"h_self{cl + 1}"] = h_self
                g.ndata[f"lstm_h{cl}"] = h_
                g.ndata[f"lstm_c{cl}"] = c_
            else:
                h_self, hist_neigh = layer(g, cur_layer=cl)
                h_self = self.activation(h_self)
                g.ndata[f"h_self{cl + 1}"] = h_self
                g.ndata[f"history_neigh{cl}"] = hist_neigh

        deg = g.ndata["history_deg"]
        g.ndata["history_deg"] = deg + g.in_degrees().to(h_self)    
        return g

class LinkLayer(nn.Module):
    def __init__(self, in_features=128, out_features=1, concat=True, time_encoding="concat", dropout=0.2, proj=True):
        super(LinkLayer, self).__init__()
        self.concat = concat
        mul = 2 if concat else 1
        self.time_encoder = TimeEncodingLayer(
            in_features, in_features, time_encoding=time_encoding)
        self.fc = nn.Linear(in_features * mul, out_features)
        self.dropout = nn.Dropout(dropout)
        self.proj = proj
    
    def forward(self, src_emb, dst_emb, src_t, dst_t, t):
        emb_u = self.time_encoder(src_emb, t - src_t)
        emb_v = self.time_encoder(dst_emb, t - dst_t)
        if self.concat:
            x = torch.cat([emb_u, emb_v], dim=1)
        else:
            x = emb_u + emb_v
        logits = self.fc(self.dropout(x))
        return logits.squeeze()

class OnlineGTC(nn.Module):
    def __init__(self, g, in_feats, edge_feats, n_hidden, args) -> None:
        super(OnlineGTC, self).__init__()
        self.nfeat = g.ndata["nfeat"]
        self.efeat = g.edata["efeat"]
        self.logger = logging.getLogger()
        self.conv = OnlineSAGE(in_feats, n_hidden, edge_feats, args.n_layers,
                               F.relu, args.dropout, args.agg_type)
        self.pred = LinkLayer(n_hidden,
                              1,
                              time_encoding=args.time_encoding,
                              proj=args.projection)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.n_layers = args.n_layers
        self.n_neg = args.n_neg

        if args.norm:
            self.norm = nn.LayerNorm(n_hidden)
        else:
            self.norm = None

    def forward(self, g, batch_samples):
        g = g.local_var()
        # update h_self{0|...|layers}
        # update node_last_time
        device = g.ndata["nfeat"].device
        batch_samples = [torch.tensor(s).to(device) for s in batch_samples]
        src, dst, t = batch_samples
        t = t.float()

        h_nodes = g.ndata[f"h_self{self.n_layers}"]
        if self.norm is not None:
            h_nodes = self.norm(h_nodes)
        h_time = g.ndata["last_time"]
        logits = self.pred(h_nodes[src], h_nodes[dst], h_time[src], h_time[dst], t)
        return logits.sigmoid()

    def update(self, graph, batch_eids):
        g = graph.local_var()
        # edges involved nodes
        sg = g.edge_subgraph(batch_eids)
        # dgl 0.4.3post2 doesn't preserve the requires_grad as True when nfeat
        # is trainable.
        sg.copy_from_parent()
        sg.ndata["nfeat"] = g.ndata["nfeat"][sg.ndata[dgl.NID]]
        # compute new_neigh features
        new_sg = self.conv(sg)
        # update features
        new_sg.copy_to_parent()

        graph.ndata["last_time"] = g.ndata["last_time"]
        graph.ndata["history_deg"] = g.ndata["history_deg"]
        graph.ndata["h_self0"] = g.ndata["h_self0"]
        for i in range(self.n_layers):
            graph.ndata[f"history_neigh{i}"] = g.ndata[f"history_neigh{i}"]
            graph.ndata[f"lstm_h{i}"] = g.ndata[f"lstm_h{i}"] 
            graph.ndata[f"lstm_c{i}"] =  g.ndata[f"lstm_c{i}"]
            graph.ndata[f"h_self{i + 1}"] = g.ndata[f"h_self{i + 1}"]


def init_graph(g, n_layers):
    nfeat = g.ndata["nfeat"]
    # Initialization.
    for i in range(n_layers):
        g.ndata[f"history_neigh{i}"] = torch.zeros_like(nfeat)
        g.ndata[f"lstm_h{i}"] = torch.zeros_like(nfeat)
        g.ndata[f"lstm_c{i}"] = torch.zeros_like(nfeat)
    for i in range(n_layers):
        g.ndata[f"h_self{i + 1}"] = torch.zeros_like(nfeat)

    g.ndata["last_time"] = torch.zeros(nfeat.shape[0]).to(nfeat)
    g.ndata["history_deg"] = torch.zeros(nfeat.shape[0]).to(nfeat)


def align_data_with_graph(g, val_labels):
    # Call graph ndata, edata to forece device move.
    cpu_g = dgl.graph(g.edges())
    for k in g.ndata.keys():
        cpu_g.ndata[k] = g.ndata[k].to("cpu")
    for k in g.edata.keys():
        cpu_g.edata[k] = g.edata[k].to("cpu")

    val_data = TemporalDataset(cpu_g, val_labels, train=False)
    val_samples = next(
        iter(
            DataLoader(val_data,
                       batch_size=len(val_labels),
                       shuffle=False,
                       num_workers=0)))
    return val_samples
    

@torch.no_grad()
def eval_fastgtc(model, g, batch_samples):
    logits = model.infer(g, batch_samples)
    logits = logits.sigmoid().cpu().numpy()
    return logits


@torch.no_grad()
def eval_online(model, g, val_labels):
    # We update the edges in graph chronologically before each sample in 
    # val_labels.
    src_l = val_labels["from_node_id"].to_numpy()[:, np.newaxis]
    dst_l = val_labels["to_node_id"].to_numpy()[:, np.newaxis]
    ts_l = val_labels["timestamp"].to_numpy()[:, np.newaxis]
    t_th = g.edata["timestamp"].cpu().numpy()
    start_eid = 0
    end_eid = 1
    logits = []
    # Init before val_labels.
    ts_train = np.unique(t_th[t_th < ts_l.min()])
    for idx in trange(len(ts_train)):
        end_eid = np.searchsorted(t_th, ts_train[idx], side="left")
        if start_eid < end_eid:
            model.update(g, np.arange(start_eid, end_eid))
            start_eid = end_eid

            if False:
                # Check the equality between FastGTC and OnlineGTC after each update.
                n_layer = model.n_layers + 1
                h_self = [g.ndata[f"h_self{i}"] for i in range(n_layer)]
                
                tg = g.edge_subgraph(np.arange(end_eid), preserve_nodes=True)
                tg.copy_from_parent()
                def wrapper(layer):
                    def simple_reduce(nodes): 
                        return {f"dst_feat{layer}": 
                            nodes.mailbox[f"m{layer}"][:, -1, :].squeeze(1)}
                    return simple_reduce
                for i in range(n_layer):
                    tg.update_all(fn.copy_e(f"dst_feat{i}", f"m{i}"), wrapper(i))
                dst_feat = [tg.ndata[f"dst_feat{i}"] for i in range(n_layer)]

                for h_, d_ in zip(h_self, dst_feat):
                    assert torch.all(torch.abs(h_ - d_) < 1e-4)

    # Using float32 to avoid information leakage due to the float precision.
    ts_l = np.float32(ts_l)
    for idx in trange(len(src_l)):
        # Update.
        end_eid = np.searchsorted(t_th, ts_l[idx], side="left")
        if start_eid < end_eid:
            model.update(g, np.arange(start_eid, end_eid))
            start_eid = end_eid

            if False:
                # Check the equality between FastGTC and OnlineGTC after each update.
                n_layer = model.n_layers + 1
                h_self = [g.ndata[f"h_self{i}"] for i in range(n_layer)]
                
                tg = g.edge_subgraph(np.arange(end_eid), preserve_nodes=True)
                tg.copy_from_parent()
                def wrapper(layer):
                    def simple_reduce(nodes): 
                        return {f"dst_feat{layer}": 
                            nodes.mailbox[f"m{layer}"][:, -1, :].squeeze(1)}
                    return simple_reduce
                for i in range(n_layer):
                    tg.update_all(fn.copy_e(f"dst_feat{i}", f"m{i}"), wrapper(i))
                dst_feat = [tg.ndata[f"dst_feat{i}"] for i in range(n_layer)]

                for i, h_, d_ in zip(range(n_layer), h_self, dst_feat):
                    assert torch.all(torch.abs(h_ - d_) < 1e-4), i
        logits.append(model(g, (src_l[idx], dst_l[idx], ts_l[idx])).item())

        
    return np.array(logits)


@torch.no_grad()
def speed_online(model, g, val_samples, batch_size=128):
    # We assume the model has been updated.
    src_l = val_samples["from_node_id"].to_numpy()
    dst_l = val_samples["to_node_id"].to_numpy()
    ts_l = val_samples["timestamp"].to_numpy()

    n_batch = int(np.ceil(len(val_samples) / batch_size))
    start = time.time()
    for idx in trange(n_batch):
        sid = idx * batch_size
        eid = sid + batch_size
        prob = model(g, (src_l[sid: eid], dst_l[sid: eid], ts_l[sid: eid]))
        model.update(g, np.arange(sid, eid))
    end = time.time()
    return n_batch, end - start


def eval_logit(labels, logits):
    acc = accuracy_score(labels, logits >= 0.5)
    f1 = f1_score(labels, logits >= 0.5)
    auc = roc_auc_score(labels, logits)
    return acc, f1, auc


def test_online(args, logger):
    set_random_seed()
    logger.info("Set random seeds.")
    logger.info(args)

    # Set device utility.
    if args.gpu:
        if args.gid >= 0:
            device = torch.device("cuda:{}".format(args.gid))
        else:
            device = torch.device("cuda:{}".format(get_free_gpu()))
        logger.info(
            "Begin Conv on Device %s, GPU Memory %d GB", device,
            torch.cuda.get_device_properties(device).total_memory // 2**30)
    else:
        device = torch.device("cpu")

    # Load nodes, edges, and labeled dataset for training, validation and test.
    logger.info("Dataset preparation.")
    nodes, edges, train_labels, val_labels, test_labels = prepare_dataset(
        args.dataset)
    delta = edges["timestamp"].shift(-1) - edges["timestamp"]
    # Pandas loc[low:high] includes high, so we use slice operations here instead.
    assert np.all(delta[:len(delta) - 1] >= 0)

    # Set DGLGraph, node_features, edge_features, and edge timestamps.
    logger.info("Construct DGLGraph.")
    g = construct_dglgraph(edges, nodes, device, node_dim=args.n_hidden)
    t = g.edata["timestamp"]
    assert torch.all(t[1:] - t[:-1] >= 0)
    src_maxeid, dst_maxeid, src_deg, dst_deg = precompute_maxeid(g)
    g.edata["src_max_eid"] = src_maxeid.to(device)
    g.edata["dst_max_eid"] = dst_maxeid.to(device)
    g.edata["src_deg"] = src_deg.to(device)
    g.edata["dst_deg"] = dst_deg.to(device)
        
    lr = '%.4f' % args.lr

    def ckpt_path(epoch):
        return f'./ckpt/FastGTC-{args.dataset}-{args.agg_type}-{lr}-{epoch}-{args.hostname}-{device.type}-{device.index}.pth'

    MODEL_SAVE_PATH = f'./saved_models/FastGTC-{args.dataset}-{args.agg_type}-{lr}-layer{args.n_layers}-hidden{args.n_hidden}.pth'
    ckpt_state = torch.load(MODEL_SAVE_PATH)

    in_feat = g.ndata["nfeat"].shape[-1]
    edge_feat = g.edata["efeat"].shape[-1]
    gtc = FastTemporalLinkTrainer(g, in_feat, edge_feat, args.n_hidden, args)
    gtc.load_state_dict(ckpt_state)
    gtc = gtc.to(device)
    online = OnlineGTC(g, in_feat, edge_feat, args.n_hidden, args)
    online.load_state_dict(ckpt_state)
    online = online.to(device)
    
    test_samples = align_data_with_graph(g, test_labels)
    logits = eval_fastgtc(gtc, g, test_samples)
    acc, f1, auc = eval_logit(test_labels["label"], logits)
    logger.info("acc: %.3f, f1: %.3f, auc: %.3f", acc, f1, auc)

    init_graph(g, args.n_layers)
    logits = eval_online(online, g, test_labels)    
    acc, f1, auc = eval_logit(test_labels["label"], logits)
    metrics = {"accuracy": acc, "f1": f1, "auc": auc}
    write_result({"valid_auc": 0.0},
                 metrics,
                 args.dataset,
                 {},
                 postfix="OnlineGTC")
    logger.info("acc: %.3f, f1: %.3f, auc: %.3f", acc, f1, auc)

    init_graph(g, args.n_layers)
    n_batch, duration = speed_online(online, g, val_labels)
    logger.info("%d batch cost %.2f seconds.", n_batch, duration)


def train_online(args, logger):
    pass


if __name__ == "__main__":
    # Set arg_parser, logger, and etc.
    parser = fastgtc_args()
    args = parser.parse_args()
    logger = set_logger()
    test_online(args, logger)

