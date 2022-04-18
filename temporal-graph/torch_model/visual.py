import argparse
import os
from datetime import datetime

import numpy as np
import torch
from utils.util import get_free_gpu, timeit
from utils.util import set_logger, set_random_seed

from torch_model.util_dgl import construct_dglgraph

from torch_model.fast_gtc import (FastTemporalLinkTrainer, fastgtc_args,
                       precompute_maxeid, prepare_dataset)

# Change the order so that it is the one used by "nvidia-smi" and not the
# one used by all other programs ("FASTEST_FIRST")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

@timeit
def save_embeddings(args, logger):
    set_random_seed()
    logger.info(args.dataset)

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

    logger.info("Set model config.")
    # Input features: node_featurs + edge_features + time_encoding
    # in_feats = (g.ndata["nfeat"].shape[-1] + g.edata["efeat"].shape[-1])
    in_feat = g.ndata["nfeat"].shape[-1]
    edge_feat = g.edata["efeat"].shape[-1]

    model = FastTemporalLinkTrainer(g, in_feat, edge_feat, args.n_hidden, args)
    
    lr = '%.4f' % (args.lr)
    lam = '%.1f' % args.lam
    margin = '%.1f' % (args.margin)

    MODEL_SAVE_PATH = f'./saved_models/FastGTC-{args.dataset}-{args.agg_type}-{lr}-{lam}-{margin}-layer{args.n_layers}-hidden{args.n_hidden}.pth'
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model = model.to(device)
    with torch.no_grad():
        src_feat, dst_feat = model.conv(g)
    src_feat = src_feat.cpu().numpy()
    dst_feat = dst_feat.cpu().numpy()
    np.savez(f'saved_embs/{args.dataset}.npz', src_feat=src_feat, dst_feat=dst_feat)

if __name__ == "__main__":
    # Set arg_parser, logger, and etc.
    parser = fastgtc_args()
    args = parser.parse_args()
    logger = set_logger()
    fname = [
        'fb-forum', 'soc-sign-bitcoinotc', 'ia-escorts-dynamic',
        'ia-movielens-user2tags-10m', 'soc-wiki-elec', 'ia-slashdot-reply-dir'
    ] 
    for data in fname:
        args.dataset = data
        save_embeddings(args, logger)

