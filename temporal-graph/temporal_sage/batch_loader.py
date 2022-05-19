from collections.abc import Mapping
import logging
import random

import dgl
from dgl.dataloading import BlockSampler, EdgeCollator, EdgeDataLoader
import dgl.backend as F
from dgl import transform
from dgl.convert import heterograph
from dgl.dataloading.dataloader import Collator
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from IPython import embed
# from util import myout

class TemporalEdgeDataLoader(EdgeDataLoader):
    """We iterate over edges sorting by timestamp, generating the list of
    message flow graphs (MFGs) in the past time as computation dependency of
    the said minibatch.
     
    For each iteration, the object will yield similar results as 
    `EdgeDataLoader` except that timestamps of MFGs must be earlier than the
    generated minibatch edges, and we return a list of MFGs corresponding to
    history graphs.
    """
    def __init__(self, graphs, year_range, block_sampler, device='cpu', **kwargs):
        # We use full eids of each graph inside year_range for training/testing.
        # `mock_g` and `mock_eid` are only used for `EdgeDataLoader` initialization.
        mock_g = graphs[0]
        mock_eid = mock_g.edges('eid')
        super().__init__(mock_g, mock_eid, block_sampler, device=device, **kwargs)
        self.graphs = graphs
        self.year_range = year_range
        self.sampler = block_sampler
        self.device = device

        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v

        self.collator = TemporalEdgeCollator(graphs, year_range, block_sampler, **collator_kwargs)

        dataset = self.collator.dataset
        self.dataloader = DataLoader(dataset, collate_fn=self.collator.collate, **dataloader_kwargs)

    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class TemporalEdgeCollator:
    def __init__(self, graphs, year_range, block_sampler, history_window=1, negative_sampler=None):
        super().__init__()

        self.graphs = graphs
        self.year_range = year_range
        eids = [graphs[year].edges('eid') for year in year_range] # [tensor(200), ..., tensor(200)]
        years = [torch.full(eid.shape, year).int() for year, eid in zip(year_range, eids)] # [tensor(200), ..., tensor(200)]
        eids = torch.cat(eids)
        years = torch.cat(years)
        self.eids = eids # [n]
        self.years = years # [n]
        # self._dataset = torch.stack([eids, years]).t() # (N, 2)
        self._dataset = TemporalDataset(eids, years)

        self.block_sampler = block_sampler
        self.negative_sampler = negative_sampler
        self.history_window = history_window
        self.logger = logging.getLogger('TemporalEdgeCollator')
        self.warning = True
    
    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        if self.negative_sampler is None:
            return self._collate(items)
        else:
            return self._collate_with_negative_sampling(items)

    def _collate(self, items):
        # eids = items[0]
        # years = items[1]
        eids = torch.cat([item[0] for item in items]) # [b*n]
        years = torch.cat([item[1] for item in items]) # [b*n]
        min_year = years.min().item()
        assert min_year >= 1, 'Past graphs are empty.'

        if not torch.all(years == min_year) and self.warning:
            # self.logger.warning('We have to filter edges later than the \
            #     minimum year of the current batch.')
            # self.warning = False
            eids = eids[years == min_year] # [n]
            years = years[years == min_year] # [n]

        start_year = min_year - self.history_window
        history_gs = [self.graphs[i] for i in range(start_year, min_year)]
        g = self.graphs[min_year]

        self.g = g
        self.g_sampling = history_gs[-1]
        pair_graph = g.edge_subgraph(eids)
        seed_nodes = pair_graph.ndata[dgl.NID]

        history_blocks = []
        for g_sampling in history_gs:
            history_blocks.append(self.block_sampler.sample_blocks(g_sampling, seed_nodes))
        input_nodes = [blocks[0].srcdata[dgl.NID] for blocks in history_blocks]

        return input_nodes, pair_graph, history_blocks


    def _collate_with_negative_sampling(self, items):
        # items: list(tuple(tensor, tensor))
        eids = torch.stack([item[0] for item in items]) # [b]
        years = torch.stack([item[1] for item in items]) # [b]

        min_year = years.min().item()
        assert min_year >= 1, 'Past graphs are empty.'

        if not torch.all(years == min_year) and self.warning:
            # self.logger.warning('We have to filter edges later than the ' \
            #     'minimum year of the current batch.')
            # self.warning = False 
            eids = eids[years == min_year]
            years = years[years == min_year]

        start_year = min_year - self.history_window
        history_gs = [self.graphs[i] for i in range(start_year, min_year)] # [Graph(num_nodes=274, num_edges=4442)]
        g = self.graphs[min_year] # Graph(num_nodes=274, num_edges=338)

        self.g = g
        self.g_sampling = history_gs[-1]
        # pair_graph = g.edge_subgraph(eids, relabel_nodes=False) 0.7.1
        pair_graph = g.edge_subgraph(eids, preserve_nodes=True) # 0.6.1
        induced_edges = pair_graph.edata[dgl.EID] # [n]

        neg_srcdst = self.negative_sampler(g, eids) # (tensor(n), tensor(n))
        if not isinstance(neg_srcdst, Mapping):
            neg_srcdst = {g.canonical_etypes[0]: neg_srcdst} # {('_N', '_E', '_N'): (tensor(n), tensor(n))}
        dtype = F.dtype(list(neg_srcdst.values())[0][0]) # torch.int64
        ctx = F.context(pair_graph) #  device(type='cpu')
        neg_edges = {
            etype: neg_srcdst.get(etype, (F.copy_to(F.tensor([], dtype), ctx),
                                        F.copy_to(F.tensor([], dtype), ctx)))
            for etype in g.canonical_etypes
        } #  {('_N', '_E', '_N'): (tensor(n), tensor(n))}
        neg_pair_graph = heterograph(neg_edges, 
            {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}) # {'_N': 274}
        
        # Remove nodes without degrees and relabel node id.
        pair_graph, neg_pair_graph = transform.compact_graphs([pair_graph, neg_pair_graph]) 
        pair_graph.edata[dgl.EID] = induced_edges
        seed_nodes = pair_graph.ndata[dgl.NID]

        history_blocks = []
        for g_sampling in  history_gs:
            # [[Block(num_src_nodes=274, num_dst_nodes=274, num_edges=674), Block(num_src_nodes=274, num_dst_nodes=274, num_edges=491)]]
            history_blocks.append(self.block_sampler.sample_blocks(
                g_sampling, seed_nodes
            ))
        input_nodes = [blocks[0].srcdata[dgl.NID] for blocks in history_blocks]

        return input_nodes, pair_graph, neg_pair_graph, history_blocks


class TemporalDataset(Dataset):
    def __init__(self, eids, years) -> None:
        super().__init__()
        assert eids.shape == years.shape
        self.eids = eids
        self.years = years
    
    def __len__(self):
        return len(self.eids)
    
    def __getitem__(self, index):
        return self.eids[index], self.years[index]


# class TemporalEdgeCollator(EdgeCollator):
#     def __init__(self, args, g, eids, block_sampler, g_sampling=None, exclude=None,
#                  reverse_eids=None, reverse_etypes=None, negative_sampler=None, mode='val'):
#         super(TemporalEdgeCollator, self).__init__(g, eids, block_sampler, g_sampling, exclude,
#                  reverse_eids, reverse_etypes, negative_sampler)
        
#         self.args = args
#         self.mode = mode

#     def collate(self, items):
#         #print('before', self.block_sampler.ts)

#         current_ts = self.g.edata['timestamp'][items[-1]]  # only sample edges before current timestamp
#         self.block_sampler.ts = current_ts
#         neg_pair_graph = None
#         if self.negative_sampler is None:
#             input_nodes, pair_graph, blocks = self._collate(items)
#         else:
#             input_nodes, pair_graph, neg_pair_graph, blocks = self._collate_with_negative_sampling(items)

#         for i in range(self.args.n_layer-1):
#             self.block_sampler.frontiers[0].add_edges(*self.block_sampler.frontiers[i+1].edges())
#         frontier = dgl.reverse(self.block_sampler.frontiers[0])

#         return input_nodes, pair_graph, neg_pair_graph, blocks, frontier, current_ts

class MultiLayerTemporalNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, args, fanouts, replace=False, return_eids=False):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace
        self.ts = 0
        self.args = args
        self.frontiers = [None for _ in range(len(fanouts))]

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]

        g = dgl.in_subgraph(g, seed_nodes)
        g.remove_edges(torch.where(g.edata['timestamp']>self.ts)[0])

        if fanout is None:
            frontier = g
            #frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            if self.args.uniform:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            else:
                frontier = dgl.sampling.select_topk(g, fanout, 'timestamp', seed_nodes)

        self.frontiers[block_id] = frontier
        return frontier

class frauder_sampler():
    def __init__(self, g):
        self.fraud_eid = torch.where(g.edata['label']!=0)[0]
        len_frauder = self.fraud_eid.shape[0] // 2
        self.fraud_eid = self.fraud_eid[:len_frauder]
        self.ts = g.edata['timestamp'][self.fraud_eid]
    def sample_fraud_event(self, g, bs, current_ts):
        idx = (self.ts<current_ts)
        num_fraud = idx.sum().item()
        
        if num_fraud > bs:
            
            idx[random.sample(list(range(num_fraud)), num_fraud-bs)] = False # 只采样一部分fraud event
            
        fraud_eid = self.fraud_eid[idx]
        
        fraud_graph = dgl.edge_subgraph(g, fraud_eid)
        return fraud_graph
