import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class TemporalDataset(Dataset):
    def __init__(self,
                 graph,
                 edges,
                 neg_k=1,
                 train=True):
        super(TemporalDataset, self).__init__()
        device = torch.device("cpu")
        self.g = graph
        self.train_src = torch.tensor(
            edges["from_node_id"].to_numpy()).to(device)
        self.train_dst = torch.tensor(
            edges["to_node_id"].to_numpy()).to(device)
        self.train_t = torch.tensor(
            edges["timestamp"].to_numpy()).float().to(device)
        self.dst_nodes = self.train_dst.unique()
        self.n_nodes = self.dst_nodes.shape[0]

        self.in_edges = []
        self.out_edges = []
        graph_src, graph_dst = self.g.edges()
        for i in graph.nodes().sort()[0]:
            self.in_edges.append(graph.in_edges(i, 'eid').sort()[0])
            self.out_edges.append(graph.out_edges(i, 'eid').sort()[0])
            assert torch.all(graph_dst[self.in_edges[i]] == i)
            assert torch.all(graph_src[self.out_edges[i]] == i)
            assert torch.all(self.in_edges[i][1:] - self.in_edges[i][:-1] > 0)
            assert torch.all(self.out_edges[i][1:] - self.out_edges[i][:-1] > 0)

        self.neg_k = neg_k
        self.train = train

    def __getitem__(self, index: int) -> tuple:
        g = self.g
        t = g.edata["timestamp"]
        isrc, idst, it = self.train_src[index], self.train_dst[
            index], self.train_t[index]

        # src, dst, src_edges = g.out_edges(isrc, 'all')
        src_edges = self.out_edges[isrc]
        # assert torch.all(src_edges[1:] - src_edges[:-1] > 0)
        src_idx = (t[src_edges] < it).sum() - 1
        src_idx = torch.max(src_idx, torch.zeros_like(src_idx))
        src_eid = src_edges[src_idx]

        # dst_edges = g.in_edges(idst, 'eid')
        dst_edges = self.in_edges[idst]
        # print(idst)
        # print(dst_edges)
        # assert torch.all(dst_edges[1:] - dst_edges[:-1] > 0)
        dst_idx = (t[dst_edges] < it).sum() - 1
        dst_idx = torch.max(dst_idx, torch.zeros_like(dst_idx))
        dst_eid = dst_edges[dst_idx]

        # print(idst, t[dst_edges])
        # print(dst_idx, t[dst_eid], it)
        assert torch.all(torch.logical_or(t[src_eid] < it, src_idx == 0))
        assert torch.all(torch.logical_or(t[dst_eid] < it, dst_idx == 0))
        assert torch.all(t[src_eid[src_idx != 0]] < it)
        assert torch.all(t[dst_eid[dst_idx != 0]] < it)

        if not self.train:
            return it, src_eid, dst_eid

        neg_ = torch.randint(self.n_nodes, size=(self.neg_k, ))
        ineg = self.dst_nodes[neg_]
        neg_indices = torch.zeros_like(ineg)
        neg_eids = torch.zeros_like(ineg)
        for i, neg in enumerate(ineg):
            # neg_edges = g.in_edges(neg, 'eid')
            neg_edges = self.in_edges[neg]
            neg_idx = (t[neg_edges] < it).sum() - 1
            neg_idx = torch.max(neg_idx, torch.zeros_like(neg_idx))
            neg_indices[i] = neg_idx
            neg_eids[i] = neg_edges[neg_idx]

        assert torch.all(torch.logical_or(t[neg_eids] < it, neg_indices == 0))
        assert torch.all(g.has_edges_between(isrc, idst) == 1)
        return it, src_eid, dst_eid, neg_eids

    def __len__(self) -> int:
        return self.train_src.shape[0]
