'''In this script, we construct three simple classifiers accounting for three
tasks: . All of them are based on a RGCN backbone.
'''
import argparse
from asyncio.log import logger
from logging import raiseExceptions
import os
import time

import dgl
from dgl.dataloading import negative_sampler
from dgl.dataloading.neighbor import MultiLayerNeighborSampler
import dgl.function as fn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

from batch_loader import TemporalEdgeCollator, TemporalEdgeDataLoader
from batch_model import BatchModel
from model import Model, compute_loss, construct_negative_graph
from util import (CSRA_PATH, DBLP_PATH, NOTE_PATH, EarlyStopMonitor,
                  set_logger, set_random_seed, write_result)
import pickle as pkl
from IPython import embed


def _load_data(dataset="ia-contact", mode="format_data", root_dir="./"):
    edges = pd.read_csv("{}/{}/{}.edges".format(root_dir, mode, dataset))
    nodes = pd.read_csv("{}/{}/{}.nodes".format(root_dir, mode, dataset))
    return edges, nodes
    

def load_data(dataset="ia-contact", mode="format", root_dir="./"):
    """We split dataset into two files: dataset.edges, and dataset.nodes.

    """
    # Load edges and nodes dataframes from the following directories.
    # Return: a tuple of (edges, nodes) of required datasets.
    # format_data/train_data/valid_data/test_data
    # label_train_data/label_valid_data/label_test_data
    mode = "{}_data".format(mode)
    return _load_data(dataset=dataset, mode=mode, root_dir=root_dir)


def covert_to_dgl(edges, nodes):
    assert(max(edges['from_node_id'].max(), edges['to_node_id'].max()) == nodes['node_id'].max())
    assert(min(edges['from_node_id'].min(), edges['to_node_id'].min()) == nodes['node_id'].min())
    assert(np.all(np.array(nodes['id_map'].tolist()) == np.arange(len(nodes))))

    node2nid = nodes.set_index('node_id').to_dict()['id_map']
    edges['src_nid'] = edges['from_node_id'].map(node2nid)
    edges['dst_nid'] = edges['to_node_id'].map(node2nid)

    graph = dgl.graph((torch.tensor(edges['src_nid']), torch.tensor(edges['dst_nid'])))
    graph.edata['ts']  = torch.tensor(edges['timestamp'])
    graph.ndata['feat'] = torch.from_numpy(np.eye(len(nodes))).to(torch.float) # one_hot

    logger.info('Graph %s.', str(graph))
    return graph


def split_graph(args, graph, num_ts=128):
    max_ts, min_ts = graph.edata['ts'].max().item(), graph.edata['ts'].min().item()
    timespan = np.ceil((max_ts - min_ts) / num_ts)

    logger.info(f'Split graph into {num_ts} snapshots.')
    graphs = []

    if args.temporal_feat:
        spath = f'{args.root_dir}format_data/dblp-coauthors.nfeat.pkl'
        logger.info(f'Loading temporal node feature from {spath}')
        node_feats = pkl.load(open(spath, 'rb'))
        
    for i in trange(num_ts):
        ts_low = min_ts + i*timespan
        ts_high = ts_low + timespan

        eids = graph.filter_edges(lambda x: (x.data['ts']>= ts_low) & (x.data['ts'] < ts_high))
        ts_graph = graph.edge_subgraph(eids, preserve_nodes=True)

        if args.temporal_feat:
            ts_graph.ndata['feat'] = node_feats[int(ts_low)]

        if 'all' not in args.named_feats:
            feat_select = []
            old_feat = ts_graph.ndata['feat']
            for dim in args.named_feats:
                try:
                    dim = int(dim)
                except:
                    raise ValueError(f'--named_feats must be list(int), but {dim} is not a integer')
                feat_select.append(old_feat[:, dim:dim+1])
            ts_graph.ndata['feat'] = torch.hstack(feat_select)
        graphs.append(ts_graph)
    new_feat = graphs[0].ndata['feat']
    logger.info(f'Select these dims: {args.named_feats} and change node feats from {old_feat.shape} to {new_feat.shape}')
    return graphs


def train(args, model, train_loader, features, opt):
    for epoch in range(args.epochs):
        loss_avg, y_probs, y_labels = 0, [], []
        model.train()
        batch_bar = tqdm(train_loader, desc='train')
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(batch_bar):
            history_inputs = [nfeat[nodes].to(args.device) for nfeat, nodes in zip(features, input_nodes)]
            # batch_inputs = nfeats[input_nodes].to(device)
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            # blocks = [block.int().to(device) for block in blocks]
            history_blocks = [[block.int().to(args.device) for block in blocks] for blocks in history_blocks]

            pos_score, neg_score = model(history_blocks, history_inputs, pos_graph, neg_graph)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()

            y_probs.append(pos_score.detach().cpu().numpy())
            y_labels.append(np.ones_like(y_probs[-1]))
            y_probs.append(neg_score.detach().cpu().numpy())
            y_labels.append(np.zeros_like(y_probs[-1]))

            batch_bar.set_postfix(loss=round(loss.item(), 4))
        loss_avg /= len(train_loader)
        y_prob = np.hstack([y.squeeze(1) for y in y_probs])
        y_pred = y_prob > 0.5
        y_label = np.hstack([y.squeeze(1) for y in y_labels])

        acc = accuracy_score(y_label, y_pred)
        ap = average_precision_score(y_label, y_prob)
        auc = roc_auc_score(y_label, y_prob)
        f1 = f1_score(y_label, y_pred)
        logger.info('Epoch %03d Training loss: %.4f, Test ACC: %.4f, F1: %.4f, AP: %.4f, AUC: %.4f', \
            epoch, loss_avg, acc, f1, ap, auc)

    logger.info('Saving model')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)


def infer(args, model, model_path, test_loader, features):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_probs, y_labels = [], []
    y_timespan, y_outs = [], []
    min_ts, max_ts = np.inf, -np.inf
    with torch.no_grad():
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(tqdm(test_loader, desc='infer')):
            cur_ts = pos_graph.edata['ts']
            min_timespan, max_timespan = torch.min(cur_ts).item(), torch.max(cur_ts).item()
            min_ts, max_ts = min(min_ts, min_timespan), max(max_ts, max_timespan)
            if min_timespan >= args.timespan_end or max_timespan < args.timespan_start:
                continue

            history_inputs = [nfeat[nodes].to(args.device) for nfeat, nodes in zip(features, input_nodes)]
            # batch_inputs = nfeats[input_nodes].to(device)
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            # blocks = [block.int().to(device) for block in blocks]
            history_blocks = [[block.int().to(args.device) for block in blocks] for blocks in history_blocks]

            pos_score, neg_score = model(history_blocks, history_inputs, pos_graph, neg_graph)
            y_probs.append(pos_score.detach().cpu().numpy())
            y_labels.append(np.ones_like(y_probs[-1]))
            y_probs.append(neg_score.detach().cpu().numpy())
            y_labels.append(np.zeros_like(y_probs[-1]))

            y_timespan.append(cur_ts.detach().cpu().numpy())
            y_outs.append(pos_score.detach().cpu().numpy())

    if len(y_probs) == 0:
        raise ValueError(f'--timespan start/end({args.timespan_start}/{args.timespan_end}) must betweeen {min_ts} and {max_ts}.')

    y_prob = np.hstack([y.squeeze(1) for y in y_probs])
    y_pred = y_prob > 0.5
    y_label = np.hstack([y.squeeze(1) for y in y_labels])

    acc = accuracy_score(y_label, y_pred)
    ap = average_precision_score(y_label, y_prob)
    auc = roc_auc_score(y_label, y_prob)
    f1 = f1_score(y_label, y_pred)
    logger.info('Test ACC: %.4f, F1: %.4f, AP: %.4f, AUC: %.4f', acc, f1, ap, auc)

    y_ts = np.hstack([y for y in y_timespan])
    idx = np.logical_and(y_ts >= args.timespan_start, y_ts < args.timespan_end)
    y_out = np.hstack([y.squeeze(1) for y in y_outs])[idx]
    logger.info(f'Saving results({len(y_out)}/{len(y_ts)}) between timespan {args.timespan_start} and {args.timespan_end}')
    np.save(RESULE_SAVE_PATH, y_out)


def main(args):
    logger.info(f'Loading dataset {args.dataset} from {args.root_dir}')
    edges, nodes = load_data(dataset=args.dataset, mode="format", root_dir=args.root_dir)
    graph = covert_to_dgl(edges, nodes)
    coauthors = split_graph(args, graph, num_ts=args.num_ts)

    node_features = coauthors[0].ndata['feat']
    n_features = node_features.shape[1]

    model = BatchModel(n_features, args.n_hidden, args.embed_dim, args.n_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters())

    train_idx = int(len(coauthors) * 0.75)
    features = [g.ndata['feat'] for g in coauthors]
    num_nodes = coauthors[0].number_of_nodes()
    num_edges = sum([g.number_of_edges() for g in coauthors])

    sampler = MultiLayerNeighborSampler([15, 10])
    neg_sampler = negative_sampler.Uniform(5)
    train_range = list(range(1, train_idx))
    test_range = list(range(train_idx, len(coauthors)))
    train_loader = TemporalEdgeDataLoader(coauthors, train_range, 
        sampler, negative_sampler=neg_sampler, batch_size=args.bs, shuffle=False,
        drop_last=False, num_workers=0)
    test_loader = TemporalEdgeDataLoader(coauthors, test_range,
        sampler, negative_sampler=neg_sampler, batch_size=args.bs, shuffle=False,
        drop_last=False, num_workers=0)

    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    train(args, model, train_loader, features, opt)
    infer(args, model, MODEL_SAVE_PATH, test_loader, features)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", 
                        type=str, 
                        default="ia-contact", 
                        help="dataset name", \
        choices=['fb-forum', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees', 'ia-escorts-dynamic', \
            'ia-movielens-user2tags-10m', 'ia-primary-school-proximity', 'ia-radoslaw-email', 'ia-reality-call', \
                'ia-retweet-pol', 'ia-slashdot-reply-dir', 'ia-workplace-contacts', 'soc-sign-bitcoinotc', \
                    'soc-wiki-elec', 'dblp-coauthors'])
    parser.add_argument('--root_dir', 
                        type=str, 
                        default='./')
    parser.add_argument('--prefix',
                        type=str,
                        default='TemporalSAGE',
                        help='prefix to name the checkpoints')
    parser.add_argument("--epochs", 
                        type=int, 
                        default=50,
                        help="number of training epochs")
    parser.add_argument("--bs", 
                        type=int, 
                        default=1024,
                        help="batch_size")
    parser.add_argument("--num_ts", 
                        type=int, 
                        default=128,
                        help="nums of snapshot to split")
    parser.add_argument("--n_hidden", 
                        type=int, 
                        default=100,
                        help="number of hidden units")
    parser.add_argument("--embed_dim", 
                        type=int, 
                        default=100,
                        help="dimension of node embedings")
    parser.add_argument("--n_layers", 
                        type=int, 
                        default=2,
                        help="number of propagation rounds")
    
    parser.add_argument("--gpu",
                        type=int, 
                        default=0,
                        help="gpu")
    parser.add_argument("--lr", 
                        type=float, 
                        default=1e-2,
                        help="learning rate")

    parser.add_argument('--temporal_feat', 
                        action='store_true', 
                        help='whether to use temporal node feature')
    parser.add_argument('--named_feats', 
                        nargs='+', 
                        default=['all'],
                        help='Which dimensions of features are selected for training')
    parser.add_argument('--timespan_start', 
                        type=float, 
                        default=-np.inf,
                        help='start timespan of infering')
    parser.add_argument('--timespan_end', 
                        type=float, 
                        default=np.inf,
                        help='end timespan of infering')
    
    logger = set_logger()
    set_random_seed(seed=42)
    args = parser.parse_args()
    logger.info(args)

    PARAM_STR = f'{args.epochs}-{args.bs}-{args.num_ts}-{args.n_hidden}'
    PARAM_STR += f'-{args.embed_dim}-{args.n_layers}-{args.lr}'
    PARAM_STR += f'-{args.temporal_feat}-{args.named_feats}'

    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{PARAM_STR}-{args.dataset}.pth'
    RESULE_SAVE_PATH = f'./saved_models/{args.prefix}-{PARAM_STR}-{args.timespan_start}-{args.timespan_end}-{args.dataset}.npy'
    args.device = torch.device('cuda:{}'.format(args.gpu))
    
    main(args)