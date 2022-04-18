import argparse
from datetime import datetime
import logging
import os
import random

from torch.utils.data.dataloader import DataLoader
# from torch_model.eid_precomputation import LatestNodeInteractionFinder

import dgl
from numba import jit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_loader.data_util import load_data, load_split_edges, load_label_edges
from utils.util import get_free_gpu, timeit, EarlyStopMonitor
from torch_model.util_dgl import construct_dglgraph
from torch_model.layers import TemporalLinkLayer, FastTSAGEConv, TimeEncode, TimeEncodingLayer
from torch_model.dataset import TemporalDataset
from utils.util import set_logger, set_random_seed, write_result
# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


class TGraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 edge_feats,
                 n_layers,
                 activation,
                 dropout,
                 agg_type="mean",
                 time_encoding="cosine"):
        super(TGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        
        self.time_encoder = TimeEncodingLayer(in_feats + edge_feats, n_hidden,
                             time_encoding)
        for i in range(n_layers):
            self.layers.append(
                FastTSAGEConv(n_hidden,
                              n_hidden,
                              agg_type))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, g):
        """In the 1st layer, we use the node features/embeddings as the features
        for each edge. In the next layers, we store the edge features in the edges,
        named `src_feat{current_layer}` and `dst_feat{current_layer}`.
        """
        g = g.local_var()
        tfeat = g.edata["timestamp"]

        def combine_feats(edges):
            return {
                "dst_feat0":
                torch.cat([edges.dst["nfeat"], edges.data["efeat"]], dim=1)
            }

        g.apply_edges(func=combine_feats)
        dst_feat0 = self.time_encoder(g.edata["dst_feat0"], tfeat)
        src_feat0 = dst_feat0[g.edata["src_max_eid"]]
        g.edata["src_feat0"] = src_feat0
        g.edata["dst_feat0"] = dst_feat0

        for i, layer in enumerate(self.layers):
            cl = i + 1
            dst_feat = layer(g, current_layer=cl)
            dst_feat = self.activation(self.dropout(dst_feat))
            src_feat = dst_feat[g.edata["src_max_eid"]]

            g.edata[f"src_feat{cl}"] = src_feat
            g.edata[f"dst_feat{cl}"] = dst_feat

        l = len(self.layers)
        src_feat, dst_feat = g.edata[f"src_feat{l}"], g.edata[f"dst_feat{l}"]
        return src_feat, dst_feat


class FastTemporalLinkTrainer(nn.Module):
    def __init__(self, g, in_feats, edge_feats, n_hidden, args):
        super(FastTemporalLinkTrainer, self).__init__()
        self.nfeat = g.ndata["nfeat"]
        self.efeat = g.edata["efeat"]
        self.logger = logging.getLogger()
        self.logger.info("nfeat: %r, efeat: %r", self.nfeat.requires_grad,
                         self.efeat.requires_grad)

        self.conv = TGraphSAGE(in_feats, n_hidden, edge_feats, args.n_layers,
                               F.relu, args.dropout, args.agg_type,
                               args.time_encoding)
        self.pred = TemporalLinkLayer(n_hidden,
                                      1,
                                      time_encoding=args.time_encoding,
                                      proj=args.projection)
       
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.n_neg = args.n_neg

        if args.norm:
            self.norm = nn.LayerNorm(n_hidden)
        else:
            self.norm = None

    def forward(self, g, batch_samples):
        g = g.local_var()
        device = g.ndata["nfeat"].device
        batch_samples = [s.to(device) for s in batch_samples]
        t, src, dst, neg = batch_samples
        t = t.float()
        neg = neg.flatten()

        src_feat, dst_feat = self.conv(g)
        if self.norm is not None:
            src_feat, dst_feat = self.norm(src_feat), self.norm(dst_feat)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat

        pos_logits = self.pred(g, src, dst, t)
        neg_logits = self.pred(g, src.repeat(self.n_neg), neg,
                               t.repeat(self.n_neg))
        
        loss = self.loss_fn(pos_logits, torch.ones_like(pos_logits))
        loss += self.loss_fn(neg_logits, torch.zeros_like(neg_logits))

        return loss, pos_logits, neg_logits

    def infer(self, g, batch_samples):
        self.eval()
        g = g.local_var()
        device = g.ndata["nfeat"].device
        batch_samples = [s.to(device) for s in batch_samples]
        src_feat, dst_feat = self.conv(g)
        if self.norm is not None:
            src_feat, dst_feat = self.norm(src_feat), self.norm(dst_feat)
        g.edata["src_feat"] = src_feat
        g.edata["dst_feat"] = dst_feat

        device = self.nfeat.device
        t, u, v = batch_samples
        t = t.float()
        logits = self.pred(g, u, v, t)
        return logits


def prepare_dataset(dataset):
    train, val, test, nodes = load_split_edges(dataset=dataset)
    edges = pd.concat([train, val, test]).reset_index(drop=True)
    train_labels, val_labels, test_labels, _ = load_label_edges(dataset=dataset)
    id2idx = {row.node_id: row.id_map for row in nodes.itertuples()}

    def _f(edges):
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        return edges

    edges, train_labels, val_labels, test_labels = [
        _f(e) for e in [edges, train_labels, val_labels, test_labels]
    ]
    tmax, tmin = edges["timestamp"].max(), edges["timestamp"].min()

    def scaler(s):
        return (s - tmin) / (tmax - tmin)

    # def scaler(s): return (s - tmin)
    edges["timestamp"] = scaler(edges["timestamp"])
    train_labels["timestamp"] = scaler(train_labels["timestamp"])
    val_labels["timestamp"] = scaler(val_labels["timestamp"])
    test_labels["timestamp"] = scaler(test_labels["timestamp"])
    return nodes, edges, train_labels, val_labels, test_labels


# @jit
def _par_maxeid(src, t, offset_l, in_eids, t_edges):
    src_deg = np.zeros_like(src)
    src_maxeid = np.zeros_like(src)
    for i in range(len(src)):
        isrc, it = src[i], t[i]
        src_eids = in_eids[offset_l[isrc]:offset_l[isrc + 1]]
        src_t = t_edges[src_eids]
        right = np.searchsorted(src_t, it, side='right')
        src_deg[i] = right
        src_maxeid[i] = src_eids[right - 1]
    return src_deg, src_maxeid


# @timeit
def precompute_maxeid(graph):
    """ To save gpu memory, we only compute the embedding for dst nodes at each
        layer, i.e., `dst_feat`. Thus, we get the src nodes' embeddings by the
        indices of their corresponding dst nodes.
    """
    g = graph.local_var()
    ts = g.edata["timestamp"].cpu()
    src, dst, eids = g.edges('all')

    in_edges = []
    assert torch.all(g.nodes()[1:] - g.nodes()[:-1] > 0)
    for i in g.nodes().sort()[0]:
        in_edges.append(g.in_edges(i, 'eid').sort()[0])
        assert torch.all(in_edges[i][1:] - in_edges[i][:-1] > 0)

    in_eids = [in_edges[i].numpy() for i in g.nodes()]
    offset_l = np.cumsum([0] + [len(e) for e in in_eids])
    in_eids = np.concatenate(in_eids)
    src_np, dst_np, ts_np = src.numpy(), dst.numpy(), ts.numpy()

    src_deg, src_maxeid = _par_maxeid(src_np, ts_np, offset_l, in_eids, ts_np)
    src_deg = torch.tensor(src_deg).to(src)
    src_maxeid = torch.tensor(src_maxeid).to(src)

    dst_deg, dst_maxeid = _par_maxeid(dst_np, ts_np, offset_l, in_eids, ts_np)
    dst_deg = torch.tensor(dst_deg).to(dst)
    dst_maxeid = torch.tensor(dst_maxeid).to(dst)

    assert torch.all(dst[src_maxeid] == src).item()
    assert torch.all(dst[dst_maxeid] == dst).item()
    assert torch.all(ts[src_maxeid] <= ts).item()
    assert torch.all(ts[dst_maxeid] <= ts).item()
    return src_maxeid, dst_maxeid, src_deg, dst_deg


@torch.no_grad()
def eval_linkpred(model, g, batch_samples, labels):
    model.eval()
    logits = model.infer(g, batch_samples)
    logits = logits.sigmoid().cpu().numpy()
    acc = accuracy_score(labels, logits >= 0.5)
    f1 = f1_score(labels, logits >= 0.5)
    auc = roc_auc_score(labels, logits)
    return acc, f1, auc


def train_fastgtc(args, logger):
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
    if not args.trainable:
        g.ndata["nfeat"] = torch.zeros_like(g.ndata["nfeat"])

    src_maxeid, dst_maxeid, src_deg, dst_deg = precompute_maxeid(g)
    g.edata["src_max_eid"] = src_maxeid.to(device)
    g.edata["dst_max_eid"] = dst_maxeid.to(device)
    g.edata["src_deg"] = src_deg.to(device)
    g.edata["dst_deg"] = dst_deg.to(device)

    logger.info("Dataset loader.")
    train_edges = train_labels[train_labels['label'] == 1]

    # Call graph ndata, edata to forece device move.
    cpu_g = dgl.graph(g.edges())
    for k in g.ndata.keys():
        cpu_g.ndata[k] = g.ndata[k].to("cpu")
    for k in g.edata.keys():
        cpu_g.edata[k] = g.edata[k].to("cpu")

    dataset = TemporalDataset(cpu_g,
                              train_edges,
                              args.n_neg,
                              train=True)
    train_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
    val_data = TemporalDataset(cpu_g, val_labels, train=False)
    val_samples = next(
        iter(
            DataLoader(val_data,
                       batch_size=len(val_labels),
                       shuffle=False,
                       num_workers=0)))
    test_data = TemporalDataset(cpu_g, test_labels, train=False)
    test_samples = next(
        iter(
            DataLoader(test_data,
                       batch_size=len(test_labels),
                       shuffle=False,
                       num_workers=0)))

    logger.info("Set model config.")
    in_feat = g.ndata["nfeat"].shape[-1]
    edge_feat = g.edata["efeat"].shape[-1]

    model = FastTemporalLinkTrainer(g, in_feat, edge_feat, args.n_hidden, args)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    # clip gradients by value: https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -args.clip, args.clip))

    # Only use positive edges, so we have to divide eids by 2.
    batch_size = args.batch_size
    num_batch = np.int(np.ceil(len(train_labels) * 0.5 / batch_size))
    epoch_bar = trange(args.epochs, disable=(not args.display))
    early_stopper = EarlyStopMonitor(max_round=5)
    for epoch in epoch_bar:
        # np.random.shuffle(train_eids)
        batch_bar = trange(num_batch, disable=(not args.display))
        for idx, batch_samples in zip(batch_bar, train_loader):
            model.train()
            optimizer.zero_grad()
            loss, pos_prob, neg_prob = model(g, batch_samples)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                pos_prob = pos_prob.cpu().detach().numpy()
                neg_prob = neg_prob.cpu().detach().numpy()
                pred_score = np.stack([pos_prob, neg_prob])
                # avoid pos_prob is a single element
                pred_score = pred_score.flatten()
                pred_label = pred_score > 0.5
                pos_label = np.ones_like(pos_prob, dtype=np.int)
                neg_label = np.zeros_like(neg_prob, dtype=np.int)
                true_label = np.stack([pos_label, neg_label])
                true_label = true_label.flatten()
                acc = accuracy_score(true_label, pred_label)
                f1 = f1_score(true_label, pred_label)
                auc = roc_auc_score(true_label, pred_score)

            batch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        acc, f1, auc = eval_linkpred(model, g, val_samples,
                                     val_labels["label"])
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.item(), acc=acc, f1=f1, auc=auc)

        lr = "%.4f" % args.lr

        def ckpt_path(epoch):
            return f'./ckpt/FastGTC-{args.dataset}-{args.agg_type}-{lr}-{epoch}-{args.hostname}-{device.type}-{device.index}.pth'

        if early_stopper.early_stop_check(auc):
            logger.info(
                f"No improvement over {early_stopper.max_round} epochs.")
            logger.info(
                f'Loading the best model at epoch {early_stopper.best_epoch}')
            model.load_state_dict(
                torch.load(ckpt_path(early_stopper.best_epoch)))
            logger.info(
                f'Loaded the best model at epoch {early_stopper.best_epoch} for inference'
            )
            break
        else:
            torch.save(model.state_dict(), ckpt_path(epoch))
    model.eval()
    _, _, val_auc = eval_linkpred(model, g, val_samples, val_labels["label"])
    acc, f1, auc = eval_linkpred(model, g, test_samples, test_labels["label"])
    params = {
        "best_epoch": early_stopper.best_epoch,
        "trainable": args.trainable,
        "opt": args.opt,
        "lr": "%.4f" % (args.lr),
        "agg_type": args.agg_type,
        "norm": args.norm,
        "n_neg": args.n_neg,
        "n_layers": args.n_layers,
        "n_hidden": args.n_hidden,
        "batch_size": args.batch_size,
        "dropout": args.dropout,
        "time_encoding": args.time_encoding,
        "proj": args.projection
    }
    metrics = {"accuracy": acc, "f1": f1, "auc": auc}
    write_result({"valid_auc": val_auc},
                 metrics,
                 args.dataset,
                 params,
                 postfix="FastGTC")
    lr = '%.4f' % args.lr
    MODEL_SAVE_PATH = f'./saved_models/FastGTC-{args.dataset}-{args.agg_type}-{lr}-layer{args.n_layers}-hidden{args.n_hidden}.pth'
    model = model.cpu()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


def fastgtc_args():
    import socket
    parser = argparse.ArgumentParser(description='Temporal GraphSAGE')
    parser.add_argument("-d", "--dataset", type=str, default="ia-contact")
    parser.add_argument("--dropout",
                        type=float,
                        default=0.2,
                        help="dropout probability")
    parser.add_argument("--log-file", action="store_true")
    parser.add_argument("--gpu",
                        dest="gpu",
                        action="store_true",
                        help="Whether use GPU.")
    parser.add_argument("--no-gpu",
                        dest="gpu",
                        action="store_false",
                        help="Whether use GPU.")
    parser.add_argument("--opt", choices=["Adam", "SGD"], default="Adam")
    hostname = socket.gethostname()
    parser.add_argument("--hostname",
                        action="store_const",
                        const=hostname,
                        default=hostname)
    parser.add_argument("--gid", type=int, default=-1, help="Specify GPU id.")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--no-trainable",
                        "-nt",
                        dest="trainable",
                        action="store_false")
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="number of training epochs")
    parser.add_argument("--time-encoding",
                        "-te",
                        type=str,
                        default="cosine",
                        help="Time encoding function.",
                        choices=["empty", "concat", "cosine", "outer"])
    parser.add_argument("--no-proj", dest="projection", action="store_false")
    parser.add_argument("-bs", "--batch-size", type=int, default=256)
    parser.add_argument("--n-hidden",
                        type=int,
                        default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers",
                        type=int,
                        default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--n-neg",
                        type=int,
                        default=1,
                        help="number of negative samples")
    parser.add_argument("--weight-decay",
                        type=float,
                        default=1e-5,
                        help="Weight for L2 loss")
    parser.add_argument("--clip",
                        type=float,
                        default=5.0,
                        help="Clip gradients by value.")
    parser.add_argument("--agg-type",
                        type=str,
                        default="gcn",
                        help="Aggregator type: mean/gcn/pool")
    parser.add_argument("--display", dest="display", action="store_true")
    parser.add_argument("--no-display", dest="display", action="store_false")
    return parser


if __name__ == "__main__":
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    warnings.filterwarnings(module='sklearn*', action='ignore', category=UndefinedMetricWarning)
    # Set arg_parser, logger, and etc.
    parser = fastgtc_args()
    args = parser.parse_args()
    logger = set_logger()
    train_fastgtc(args, logger)
