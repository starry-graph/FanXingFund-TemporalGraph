"""Unified interface to all dynamic graph model experiments"""
import argparse
import logging
import math
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from data_loader.data_util import load_data, load_graph, load_label_data
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)
from tqdm import tqdm, trange
from utils.util import (EarlyStopMonitor, RandEdgeSampler, get_free_gpu,
                        set_random_seed)

from subgraph_model.graph import SubgraphNeighborFinder
from subgraph_model.subgnn_np import SubGnnNp

#import numba


class LR(torch.nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, dim // 2)
        self.fc_2 = torch.nn.Linear(dim // 2, dim // 4)
        self.fc_3 = torch.nn.Linear(dim // 4, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=True)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


# set_random_seed()

# Argument and global variables
if True:
    parser = argparse.ArgumentParser(
        'Interface for TGAT experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='JODIE-wikipedia')
    parser.add_argument("-t", "--task", default="node", choices=["node"])
    parser.add_argument("--val_time", default=0.7, type=float)
    parser.add_argument("--node_layer", default=2, type=int)
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Only use source_node embedding or use the combined embeddings.")
    parser.add_argument('-f', '--freeze', action='store_true')
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix',
                        type=str,
                        default='Subgraph',
                        help='prefix to name the checkpoints')
    parser.add_argument('--n_degree',
                        type=int,
                        default=20,
                        help='number of neighbors to sample')
    parser.add_argument('--n_head',
                        type=int,
                        default=2,
                        help='number of heads used in attention layer')
    parser.add_argument('--n_epoch',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--n_layer',
                        type=int,
                        default=2,
                        help='number of network layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--drop_out',
                        type=float,
                        default=0.1,
                        help='dropout probability')
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help='idx for the gpu to use')
    parser.add_argument('--node_dim',
                        type=int,
                        default=120,
                        help='Dimentions of the node embedding')
    parser.add_argument('--time_dim',
                        type=int,
                        default=120,
                        help='Dimentions of the time embedding')
    parser.add_argument('--attn_mode',
                        type=str,
                        choices=['prod', 'map'],
                        default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--uniform',
                        action='store_true',
                        help='take uniform sampling from temporal neighbors')
    parser.add_argument('--num_prop', type=int, default=2)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.0)

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# Arguments
if True:
    PREFIX = args.prefix
    VAL_TIME = args.val_time
    NODE_LAYER = args.node_layer
    BALANCE = args.balance
    NEG_RATIO = args.neg_ratio
    BINARY = args.binary

    TASK = args.task
    FREEZE = args.freeze
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    
    UNIFORM = args.uniform
    ATTN_MODE = args.attn_mode
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim

    # Specific arguments
    NUM_PROP = args.num_prop
    NUM_MLP_LAYERS = args.num_mlp_layers
    ALPHA = args.alpha

    # Model initialize
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device('cuda:{}'.format(GPU))

    import socket

    DEVICE_STR = f'{socket.gethostname()}-{device.index}'
    PARAM_STR = f'{FREEZE}-{NUM_LAYER}-{NUM_HEADS}-{NUM_NEIGHBORS}'
    PARAM_STR += f'-{NUM_PROP}-{NUM_MLP_LAYERS}-{ALPHA}'
    PARAM_STR += f'-{BATCH_SIZE}-{DROP_OUT}-{UNIFORM}'

    MODEL_SAVE_PATH = f'./saved_models/{PREFIX}-{TASK}-{PARAM_STR}-{DATA}.pth'

    LR_SAVE_PATH = f'./saved_models/{PREFIX}-LR-{DATA}-{NODE_LAYER}-{BALANCE}-{NEG_RATIO}-{VAL_TIME}-nclass.pth'

    def get_checkpoint_path(epoch):
        return f'./ckpt/{PREFIX}-LR-{DATA}-{DEVICE_STR}-{NODE_LAYER}-{BALANCE}-{NEG_RATIO}-{VAL_TIME}-{epoch}-nclass.pth'


# set up logger
if True:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

# Load data and train val test split
if True:
    edges, nodes = load_data(DATA, "format")
    n_nodes = len(nodes) + 1
    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    g_df = edges[[
        "from_node_id", "to_node_id", "timestamp", "state_label"
    ]].copy()
    g_df["idx"] = np.arange(1, len(edges) + 1)
    g_df.columns = ["u", "i", "ts", "label", "idx"]
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if FREEZE:
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values
    label_l = g_df.label.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    # set_random_seed()

# set train, validation, test datasets
if True:
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

    # select validation and test dataset
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time

    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    val_e_idx_l = e_idx_l[valid_val_flag]
    val_label_l = label_l[valid_val_flag]

    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    test_e_idx_l = e_idx_l[valid_test_flag]
    test_label_l = label_l[valid_test_flag]

    val_time = np.quantile(g_df.ts, VAL_TIME)
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

# Initialize the data structure for graph and edge sampling
# build the graph for fast query
# # full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
ngh_finder = SubgraphNeighborFinder(full_adj_list,
                                    ts_l,
                                    graph_type="numpy",
                                    task=TASK,
                                    dataset=DATA,
                                    uniform=UNIFORM)

tgan = SubGnnNp(ngh_finder,
                n_feat,
                e_feat,
                n_feat_freeze=FREEZE,
                attn_mode=ATTN_MODE,
                num_layers=NUM_LAYER,
                num_prop=NUM_PROP,
                num_mlp_layers=NUM_MLP_LAYERS,
                alpha=ALPHA,
                n_head=NUM_HEADS,
                drop_out=DROP_OUT)

optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))

logger.info('loading saved TGAN model')
# model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
# tgan.load_state_dict(torch.load(model_path, map_location=device))
tgan.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start training node classification task')

lr_input = n_feat.shape[1] * (2 if BINARY else 1)
lr_model = LR(lr_input)
lr_optimizer = torch.optim.Adam(lr_model.parameters(), lr=args.lr)
lr_model = lr_model.to(device)
# tgan.ngh_finder = full_ngh_finder
idx_list = np.arange(len(train_src_l))
lr_criterion = torch.nn.BCELoss()
lr_criterion_eval = torch.nn.BCELoss()


def eval_epoch(src_l,
               dst_l,
               ts_l,
               label_l,
               batch_size,
               lr_model,
               tgan,
               num_layer=NODE_LAYER):
    pred_prob = np.zeros(len(src_l))
    loss = 0
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)
    with torch.no_grad():
        lr_model.eval()
        tgan.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance - 1, s_idx + batch_size)
            src_l_cut = src_l[s_idx:e_idx]
            dst_l_cut = dst_l[s_idx:e_idx]
            ts_l_cut = ts_l[s_idx:e_idx]
            label_l_cut = label_l[s_idx:e_idx]
            size = len(src_l_cut)

            cache_embed = embed_cache.check_cache(src_l_cut, ts_l_cut)
            if cache_embed is None:
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
                if BINARY:
                    dst_embed = tgan.tem_conv(dst_l_cut, ts_l_cut, NODE_LAYER)
                    src_embed = torch.cat([src_embed, dst_embed], dim=-1)
                embed_cache.update_cache(src_l_cut, ts_l_cut, src_embed)
            else:
                src_embed = cache_embed

            src_label = torch.from_numpy(label_l_cut).float().to(device)
            lr_prob = lr_model(src_embed).sigmoid()
            loss += lr_criterion_eval(lr_prob, src_label).item()
            pred_prob[s_idx:e_idx] = lr_prob.cpu().numpy()

    auc_roc = roc_auc_score(label_l, pred_prob)
    return auc_roc, loss / num_instance


pos_src = train_src_l[train_label_l]
pos_dst = train_dst_l[train_label_l]
pos_ts = train_ts_l[train_label_l]
pos_label = train_label_l[train_label_l]


def sample_positive(src_l_cut,
                    dst_l_cut,
                    ts_l_cut,
                    label_l_cut,
                    neg_ratio=NEG_RATIO):
    size = len(label_l_cut)
    # neg_cnt = (label_l_cut == 0).sum()
    pos_cnt = size // neg_ratio
    # if pos_cnt <= 0:
    #     return src_l_cut, dst_l_cut, ts_l_cut, label_l_cut

    max_idx = (pos_ts < ts_l_cut.max()).sum()
    idx = np.random.randint(0, max_idx, pos_cnt)

    sample_pos_src = pos_src[idx]
    sample_pos_dst = pos_dst[idx]
    sample_pos_ts = pos_ts[idx]
    sample_pos_label = pos_label[idx]

    new_src_cut = np.hstack([src_l_cut, sample_pos_src])
    new_dst_cut = np.hstack([dst_l_cut, sample_pos_dst])
    new_ts_cut = np.hstack([ts_l_cut, sample_pos_ts])
    new_label_cut = np.hstack([label_l_cut, sample_pos_label])
    return new_src_cut, new_dst_cut, new_ts_cut, new_label_cut


class EmbedCache(object):
    PRECISION = 5

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}

    def update_cache(self, src_l_cut, ts_l_cut, src_emb):
        for i, (node, ts) in enumerate(zip(src_l_cut, ts_l_cut)):
            key = (node, round(ts, self.PRECISION))
            if key not in self.cache:
                self.cache[key] = src_emb[i]

    def check_cache(self, src_l_cut, ts_l_cut):
        ans = []
        for i, (node, ts) in enumerate(zip(src_l_cut, ts_l_cut)):
            key = (node, round(ts, self.PRECISION))
            if key not in self.cache:
                return None
            ans.append(self.cache.get(key))
        return torch.stack(ans)


embed_cache = EmbedCache()

early_stopper = EarlyStopMonitor(max_round=10)
epoch_bar = trange(NUM_EPOCH)
for epoch in epoch_bar:
    lr_pred_prob = np.zeros(len(train_src_l))
    np.random.shuffle(idx_list)
    tgan = tgan.eval()
    lr_model = lr_model.train()
    #num_batch
    for k in trange(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut = train_src_l[s_idx:e_idx]
        dst_l_cut = train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]

        if BALANCE:
            src_l_cut, dst_l_cut, ts_l_cut, label_l_cut = sample_positive(
                src_l_cut, dst_l_cut, ts_l_cut, label_l_cut, NEG_RATIO)
        size = len(src_l_cut)

        lr_optimizer.zero_grad()
        with torch.no_grad():
            if epoch == 0:
                src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)
                if BINARY:
                    dst_embed = tgan.tem_conv(dst_l_cut, ts_l_cut,
                                                 NODE_LAYER)
                    src_embed = torch.cat([src_embed, dst_embed], dim=-1)

                embed_cache.update_cache(src_l_cut, ts_l_cut, src_embed)
            else:
                src_embed = embed_cache.check_cache(src_l_cut, ts_l_cut)

        src_label = torch.from_numpy(label_l_cut).float().to(device)
        lr_prob = lr_model(src_embed).sigmoid()
        lr_loss = lr_criterion(lr_prob, src_label)
        lr_loss.backward()
        lr_optimizer.step()

    # train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l, BATCH_SIZE, lr_model, tgan)
    val_auc, val_loss = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l,
                                   BATCH_SIZE, lr_model, tgan)
    epoch_bar.update()
    epoch_bar.set_postfix(val_auc=val_auc,
                          balance=BALANCE,
                          neg_ratio=NEG_RATIO)

    if early_stopper.early_stop_check(val_auc):
        break
    else:
        torch.save(lr_model.state_dict(), get_checkpoint_path(epoch))
    # train_auc, train_loss = eval_epoch(train_src_l, train_dst_l, train_ts_l, train_label_l, BATCH_SIZE, lr_model, tgan)
    # test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l, test_label_l, BATCH_SIZE, lr_model, tgan)
    # #torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
    # logger.info(f'train auc: {train_auc}, test auc: {test_auc}')

logger.info('No improvment over {} epochs, stop training'.format(
    early_stopper.max_round))
logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
best_model_path = get_checkpoint_path(early_stopper.best_epoch)
lr_model.load_state_dict(torch.load(best_model_path))
logger.info(
    f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
lr_model.eval()
torch.save(lr_model.state_dict(), LR_SAVE_PATH)

val_auc, val_loss = eval_epoch(val_src_l, val_dst_l, val_ts_l, val_label_l,
                               BATCH_SIZE, lr_model, tgan)
test_auc, test_loss = eval_epoch(test_src_l, test_dst_l, test_ts_l,
                                 test_label_l, BATCH_SIZE, lr_model, tgan)
#torch.save(lr_model.state_dict(), './saved_models/edge_{}_wkiki_node_class.pth'.format(DATA))
logger.info(f'test auc: {test_auc}')

res_path = "nc-results/{}-{}.csv".format(DATA, PREFIX)
headers = ["method", "dataset", "valid_auc", "auc", "params"]
if not os.path.exists(res_path):
    f = open(res_path, 'w+')
    f.write(",".join(headers) + "\r\n")
    f.close()
    os.chmod(res_path, 0o777)

config = f"num_prop={NUM_PROP},num_mlp_layers={NUM_MLP_LAYERS},alpha={ALPHA}"
config += f",freeze={FREEZE},binary={BINARY},node_layer={NODE_LAYER},balance={BALANCE},neg_ratio={NEG_RATIO},val_time={VAL_TIME:.2f}"
with open(res_path, "a") as file:
    file.write("{},{},{:.4f},{:.4f},\"{}\"".format(PREFIX, DATA, val_auc, test_auc,
                                                       config))
    file.write("\n")
