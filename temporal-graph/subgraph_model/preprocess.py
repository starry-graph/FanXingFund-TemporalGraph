import argparse
import os
import pickle

import dgl
import numpy as np
import numba
import pandas as pd
import torch
from tqdm import trange

from data_loader.data_util import _iterate_datasets, load_data, load_split_edges
from utils.util import timeit


def load_data_var(dataset, task="node"):
    if task == "node":
        # For node classification, we don't remove unseen nodes from the
        # training set.
        edges, nodes = load_data(dataset, mode="format")
    else:
        train_edges, val_edges, test_edges, nodes = \
        load_split_edges(dataset=dataset)
        edges = pd.concat([train_edges, val_edges, test_edges])

    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)

    pad = pd.DataFrame(columns=edges.columns)
    pad.loc[0] = [0] * len(edges.columns)
    pad = pad.astype(edges.dtypes)

    edges = pd.concat([pad, edges], axis=0).reset_index(drop=True)
    return edges


def edge2dgl(edges):
    src = edges["from_node_id"]
    dst = edges["to_node_id"]
    etime = torch.tensor(edges["timestamp"])
    efeat = torch.tensor(
        edges.iloc[:,
                   4:].to_numpy()) if len(edges.columns) > 4 else torch.zeros(
                       (len(edges), 2))

    u = np.vstack((src, dst)).transpose().flatten()
    v = np.vstack((dst, src)).transpose().flatten()
    src, dst = u, v
    etime = etime.repeat_interleave(2)
    efeat = efeat.repeat_interleave(2, dim=0)
    g = dgl.graph((src, dst))
    g.edata["timestamp"] = etime
    g.edata["efeat"] = efeat

    return g


def init_adj(edges: pd.DataFrame):
    src_l = edges["from_node_id"].to_numpy()
    dst_l = edges["to_node_id"].to_numpy()
    ts_l = edges["timestamp"].to_numpy()
    eidx_l = np.arange(len(src_l))

    max_idx = max(src_l.max(), dst_l.max())
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, eidx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    return adj_list


def init_offset(edges: pd.DataFrame):
    adj_list = init_adj(edges)
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


def subgraph_np(ngh_node_l, ngh_ts_l, ngh_eidx_l, offset_l, m=20):
    """We transform the latest M interactions into an interaction graph
     sequentially, where the node adjacency matrix is `(B, K, K)` and the
     node-edge adjacency matrix is `(B, K, M)`. To combine the node features
     and edge features, we further provide a `(B, K)` node identity vector NID,
     and a `(B, M)` edge identity vector EID. The final mesaage-passing
     functions as `h^1 = W_node * A_node * h^0[NID] + W_edge * A_edge * e[EID]`
     and `h^{l+1} = W_node * A_node * h^l + W_edge * A_edge * e[EID]`.
     We employ an attention layer for cross-layer subgraph pooling as
     `h_u = \sum_i a_{ui} h^L_{i}`, where `a_{ui}` is the attention score.
    """
    def sequence2graph(ngh_node, ngh_eidx):
        """We often make bugs when writing a lot of codes in a function.
        Thus, we move the function of transforming a sequence into a graph
        in a internal function. The inputs are all (M,) arrays, and we return
        two adjacency matrices, node2node and edge2node, and two identity
        vectors, containing original ids of nodes and edges.
        """
        num_nodes = ngh_eidx.shape[0]
        num_edges = ngh_eidx.shape[0]
        node2node = np.eye(num_nodes)  # self-loop
        # node2node = np.zeros((num_nodes, num_nodes))
        edge2node = np.zeros((num_nodes, num_edges))
        node_ids = np.unique(ngh_node)
        inv_nid = {nid: idx for idx, nid in enumerate(node_ids)}
        edge_ids = ngh_eidx
        inv_eid = {eid: idx for idx, eid in enumerate(edge_ids)}

        for src, dst in zip(ngh_node[:-1],
                            ngh_node[1:]):  # sequential node to ndoe
            src_idx, dst_idx = inv_nid[src], inv_nid[dst]
            node2node[dst_idx, src_idx] = 1.0
            # node2node[src_idx, dst_idx] = 1.0 # inverse link
        for dst, edge in zip(ngh_node, ngh_eidx):  # edge to node
            dst_idx, eidx = inv_nid[dst], inv_eid[edge]
            edge2node[dst_idx, eidx] = 1.0
        return node2node, node_ids, edge2node, edge_ids

    # padding node 0, and padding edge 0
    k = m # m interactions have at most m neighbors
    adj_n2n = []  # (E, K, K)
    adj_nids = []  # (E, K)
    adj_e2n = []  # (E, K, M)
    adj_eids = []  # (E, M)
    for nid in trange(len(offset_l) - 1):
        start, end = offset_l[nid], offset_l[nid + 1]
        for j in range(start, end):
            left = max(start, j - m + 1)
            right = j + 1
            ngh_node = ngh_node_l[left:right]
            ngh_eidx = ngh_eidx_l[left:right]
            n2n, nids, e2n, eids = sequence2graph(ngh_node, ngh_eidx)

            row_n2n = np.zeros((k, k))
            row_nids = np.zeros((k,), dtype=np.int)
            row_e2n = np.zeros((k, m))
            row_eids = np.zeros((m,), dtype=np.int)
            
            row_n2n[:len(n2n), :len(n2n)] = n2n
            row_nids[:len(nids)] = nids
            row_e2n[:e2n.shape[0], :e2n.shape[1]] = e2n
            row_eids[:len(eids)] = eids

            adj_n2n.append(row_n2n)
            adj_nids.append(row_nids)
            adj_e2n.append(row_e2n)
            adj_eids.append(row_eids)

    mat_n2n = np.stack(adj_n2n, axis=0)
    mat_nids = np.stack(adj_nids, axis=0)
    mat_e2n = np.stack(adj_e2n, axis=0)
    mat_eids = np.stack(adj_eids, axis=0)
    
    assert len(mat_n2n) == len(ngh_node_l)
    return mat_n2n, mat_nids, mat_e2n, mat_eids


def subgraph_dgl(ngh_l, nts_l, eidx_l, offset_l, m=20):
    # padding node 0, and padding edge 0
    adj_subgraphs = []
    for nid in trange(len(offset_l) - 1):
        start, end = offset_l[nid], offset_l[nid + 1]
        for j in range(start, end):
            # We transform these interactions into two adjacency matrices,
            # along with two identity vectors.
            slice_ngh = np.arange(max(0, j - m + 1), j + 1)
            row_nid = np.unique(ngh_l[slice_ngh])
            # We allow parallel edges in edge2node matrix.
            row_eid = eidx_l[slice_ngh][1:]

            if len(slice_ngh) > 1:  # ensure >= 2 interactions
                inv_nid = {ngh_id: idx for idx, ngh_id in enumerate(row_nid)}
                src, dst = ngh_l[slice_ngh][:-1], ngh_l[slice_ngh][1:]
                map_src = [inv_nid[sid] for sid in src]
                map_dst = [inv_nid[did] for did in dst]
                sg = dgl.graph((map_src, map_dst))
                sg.ndata[dgl.NID] = torch.LongTensor(row_nid)
                sg.edata[dgl.EID] = torch.LongTensor(row_eid)
            else:
                sg = dgl.graph((row_nid, row_nid))
                sg.ndata[dgl.NID] = torch.LongTensor(row_nid)
                sg.edata[dgl.EID] = torch.LongTensor(eidx_l[slice_ngh])
            adj_subgraphs.append(sg)
    return adj_subgraphs


@timeit
def interaction2subgraph(dataset, m=20, task="edge", dgl_type=False):
    """For each (node, timestamp) pair, we construct an interaction graph,
     which contains the latest `k` neighbors and `m` interactions, resuling in 
     a `k*k` adjacency matrix. We store the graph in a adajacency list due to
     the potential parallel edge between nodes. The requirement for a fixed
     number of nodes is for the convenience of graph neural networks.

     The graph is padded with `(0, 0)` interaction at time `0`. For nodes with
     less than `k` neighbors, we leave all other nodes as the padding node `0`.
    """

    edges = load_data_var(dataset, task)
    ngh_l, nts_l, eidx_l, offset_l = init_offset(edges)

    if dgl_type:
        adj_subgraphs = subgraph_dgl(ngh_l, nts_l, eidx_l, offset_l, m)
        path = f"subgraph_cache/{task}-{dataset}-{m}.dgl"
        # pickle.dump(adj_subgraphs, open(path, "wb"))
        dgl.save_graphs(path, adj_subgraphs)
    else:
        mat_n2n, mat_nids, mat_e2n, mat_eids = subgraph_np(
            ngh_l, nts_l, eidx_l, offset_l, m)
        path = f"subgraph_cache/{task}-{dataset}-{m}.npz"
        np.savez(path, mat_n2n=mat_n2n, mat_nids=mat_nids,
                mat_e2n=mat_e2n, mat_eids=mat_eids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper for preprocessing interaction subgraphs.")
    parser.add_argument("-t", "--task", default="edge", choices=["edge", "node"])
    parser.add_argument("-dt", "--dgl-type", action="store_true")
    parser.add_argument("-m", "--m", default=20, type=int)
    args = parser.parse_args()

    if args.task == "node":
        datasets = ["JODIE-wikipedia", "JODIE-mooc", "JODIE-reddit"]
    else:
        datasets = _iterate_datasets()[:13]

    for data in datasets:
        print(data)
        interaction2subgraph(data, args.m, args.task, args.dgl_type)