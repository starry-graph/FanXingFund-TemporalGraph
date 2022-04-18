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

# set_random_seed()

# Argument and global variables
if True:
    parser = argparse.ArgumentParser(
        'Interface for TGAT experiments on link predictions')
    parser.add_argument('-d',
                        '--data',
                        type=str,
                        help='data sources to use',
                        default='ia-contact')
    parser.add_argument("-t", "--task", default="edge", choices=["edge"])
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
    parser.add_argument('--num_prop', type=int, default=3)
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
    PARAM_STR = f'{NUM_LAYER}-{NUM_HEADS}-{NUM_NEIGHBORS}'
    PARAM_STR += f'-{NUM_PROP}-{NUM_MLP_LAYERS}-{ALPHA}'
    PARAM_STR += f'-{BATCH_SIZE}-{DROP_OUT}-{UNIFORM}'

    MODEL_SAVE_PATH = f'./saved_models/{PREFIX}-{TASK}-{PARAM_STR}-{DATA}.pth'

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
    edges, n_nodes, val_time, test_time = load_graph(DATA)
    g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
    g_df["idx"] = np.arange(1, len(g_df) + 1)
    g_df.columns = ["u", "i", "ts", "idx"]

    # if len(edges.columns) > 4:
    e_feat = edges.iloc[:, 4:].to_numpy()
    padding = np.zeros((1, e_feat.shape[1]))
    e_feat = np.concatenate((padding, e_feat))
    # else:
    #     e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if FREEZE:
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values
    # label_l = g_df.label.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    # set_random_seed()

# set train, validation, test datasets
if True:
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]

    train_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_sampler = RandEdgeSampler(src_l, dst_l)
    test_sampler = RandEdgeSampler(src_l, dst_l)

# set train, validation, test datasets
if True:
    _, val_data, test_data = load_label_data(dataset=DATA)

    val_src_l = val_data.u.values
    val_dst_l = val_data.i.values
    val_ts_l = val_data.ts.values
    val_label_l = val_data.label.values

    test_src_l = test_data.u.values
    test_dst_l = test_data.i.values
    test_ts_l = test_data.ts.values
    test_label_l = test_data.label.values

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

num_instance = len(test_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))

logger.info('loading saved TGAN model')
# model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
# tgan.load_state_dict(torch.load(model_path, map_location=device))
tgan.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start computing attention weights task')

attention_list = []
def forward_hook(module, input, output):
    ans, weights = output
    attention_list.append(weights.detach().cpu().numpy())

last_fusion_layer = tgan.fusion_layer[-1]
last_fusion_layer.register_forward_hook(forward_hook)

tgan = tgan.eval()
#num_batch
for k in trange(num_batch):
    s_idx = k * BATCH_SIZE
    e_idx = min(num_instance, s_idx + BATCH_SIZE)
    src_l_cut = test_src_l[s_idx:e_idx]
    dst_l_cut = test_dst_l[s_idx:e_idx]
    ts_l_cut = test_ts_l[s_idx:e_idx]
    label_l_cut = test_label_l[s_idx:e_idx]

    size = len(src_l_cut)

    with torch.no_grad():
        src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, NODE_LAYER)

print('num_batch: ', len(attention_list), 'weight shape: ', attention_list[0].shape) 
attention_arr = np.concatenate(attention_list)
np.savez('subgraph_cache/{}'.format(DATA), weights=attention_arr)