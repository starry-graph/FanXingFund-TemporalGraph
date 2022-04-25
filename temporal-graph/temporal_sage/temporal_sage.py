'''In this script, we construct three simple classifiers accounting for three
tasks: . All of them are based on a RGCN backbone.
'''
import argparse
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
from tqdm import tqdm

from batch_loader import TemporalEdgeCollator, TemporalEdgeDataLoader
from batch_model import BatchModel
from model import Model, compute_loss, construct_negative_graph
from util import (CSRA_PATH, DBLP_PATH, NOTE_PATH, EarlyStopMonitor,
                  set_logger, set_random_seed, write_result)


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
    graph.edata['ts']  = torch.tensor(edges['timestamp']).unsqueeze(1)
    graph.ndata['feat'] = torch.from_numpy(np.eye(len(nodes))).to(torch.float) # one_hot

    logger.info('Graph %s.', str(graph))
    return graph


def split_graph(graph, num_ts=128):
    max_ts, min_ts = graph.edata['ts'].max(), graph.edata['ts'].min()
    timespan = (max_ts - min_ts) // num_ts + 1

    logger.info(f'Split graph into {num_ts} snapshots.')
    graphs = []
    for i in range(num_ts):
        ts_low = min_ts + i*timespan
        ts_high = ts_low + timespan

        eids = graph.filter_edges(lambda x: (x.data['ts']>= ts_low) & (x.data['ts'] < ts_high))
        ts_graph = graph.edge_subgraph(eids, preserve_nodes=True)
        graphs.append(ts_graph)
    return graphs


datasets = ['fb-forum', 'ia-contact', 'ia-contacts_hypertext2009', 'ia-enron-employees', 'ia-escorts-dynamic', \
    'ia-movielens-user2tags-10m', 'ia-primary-school-proximity', 'ia-radoslaw-email', 'ia-reality-call', \
        'ia-retweet-pol', 'ia-slashdot-reply-dir', 'ia-workplace-contacts', 'soc-sign-bitcoinotc', 'soc-wiki-elec']
root_dir = './'

def main(args):
    logger.info(f'Loading dataset {args.dataset} from {root_dir}.')
    edges, nodes = load_data(dataset=args.dataset, mode="format", root_dir=root_dir)
    graph = covert_to_dgl(edges, nodes)
    coauthors = split_graph(graph, num_ts=args.num_ts)

    device = torch.device('cuda:{}'.format(args.gpu))
    # coauthors = [g.to(device) for g in coauthors]
    node_features = coauthors[0].ndata['feat']
    n_features = node_features.shape[1]

    model = BatchModel(n_features, 100, 100).to(device)
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
        sampler, negative_sampler=neg_sampler, batch_size=1024, shuffle=False,
        drop_last=False, num_workers=0)
    test_loader = TemporalEdgeDataLoader(coauthors, test_range,
        sampler, negative_sampler=neg_sampler, batch_size=1024, shuffle=False,
        drop_last=False, num_workers=0)

    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    for epoch in range(50):
        loss_avg = 0

        model.train()
        batch_bar = tqdm(train_loader)
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(batch_bar):
            history_inputs = [nfeat[nodes].to(device) for nfeat, nodes in zip(features, input_nodes)]
            # batch_inputs = nfeats[input_nodes].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            # blocks = [block.int().to(device) for block in blocks]
            history_blocks = [[block.int().to(device) for block in blocks] for blocks in history_blocks]

            pos_score, neg_score = model(history_blocks, history_inputs, pos_graph, neg_graph)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()
            batch_bar.set_postfix(loss=round(loss.item(), 4))

        loss_avg /= len(train_loader)
        y_probs = []
        y_labels = []
        model.eval()
        with torch.no_grad():
            for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(tqdm(test_loader)):
                history_inputs = [nfeat[nodes].to(device) for nfeat, nodes in zip(features, input_nodes)]
                # batch_inputs = nfeats[input_nodes].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                # blocks = [block.int().to(device) for block in blocks]
                history_blocks = [[block.int().to(device) for block in blocks] for blocks in history_blocks]

                pos_score, neg_score = model(history_blocks, history_inputs, pos_graph, neg_graph)
                y_probs.append(pos_score.detach().cpu().numpy())
                y_labels.append(np.ones_like(y_probs[-1]))
                y_probs.append(neg_score.detach().cpu().numpy())
                y_labels.append(np.zeros_like(y_probs[-1]))

        y_probs = [y.squeeze(1) for y in y_probs]
        y_labels = [y.squeeze(1) for y in y_labels]
        y_prob = np.hstack(y_probs)
        y_pred = np.hstack(y_probs) > 0.5
        y_label = np.hstack(y_labels)
        ap = average_precision_score(y_label, y_prob)
        auc = roc_auc_score(y_label, y_prob)
        f1 = f1_score(y_label, y_pred)
        logger.info('Epoch %03d Training loss: %.4f, Test F1: %.4f, AP: %.4f, AUC: %.4f', epoch, loss_avg, f1, ap, auc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dataset", type=str, default="ia-contact",
            help="dataset name")
    parser.add_argument("--num_ts", type=int, default=128,
            help="nums of snapshot to split")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")

    logger = set_logger()
    set_random_seed(seed=42)
    args = parser.parse_args()
    logger.info(args)
    main(args)
