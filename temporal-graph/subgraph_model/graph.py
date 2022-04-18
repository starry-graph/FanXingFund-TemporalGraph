import argparse
import dgl
import logging
import math
from numba import jit
import numpy as np
import os
import torch
import time
from tqdm import trange

from subgraph_model.preprocess import load_data_var, init_adj, interaction2subgraph, subgraph_np, subgraph_dgl


@jit
def find_before_nb(src_idx, cut_time, off_set_l, node_idx_l, node_ts_l,
                   edge_idx_l):
    neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

    right = np.searchsorted(neighbors_ts, cut_time, side="left")
    return neighbors_idx[:right], neighbors_e_idx[:right], neighbors_ts[:right]


@jit
def get_temporal_neighbor_nb(src_idx_l,
                             cut_time_l,
                             num_neighbors,
                             off_set_l,
                             node_idx_l,
                             node_ts_l,
                             edge_idx_l,
                             uniform=True):
    assert (len(src_idx_l) == len(cut_time_l))

    out_ngh_node_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)
    out_ngh_t_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.float32)
    out_ngh_eidx_batch = np.zeros(
        (len(src_idx_l), num_neighbors)).astype(np.int32)

    for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
        ngh_idx, ngh_eidx, ngh_ts = find_before_nb(src_idx, cut_time,
                                                   off_set_l, node_idx_l,
                                                   node_ts_l, edge_idx_l)

        if len(ngh_idx) > 0:
            if uniform:
                sampled_idx = np.random.randint(0, len(ngh_idx), num_neighbors)

                out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
                out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
                out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]

                # resort based on time
                pos = out_ngh_t_batch[i, :].argsort()
                out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
                out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
                out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]
            else:
                ngh_ts = ngh_ts[-num_neighbors:]
                ngh_idx = ngh_idx[-num_neighbors:]
                ngh_eidx = ngh_eidx[-num_neighbors:]

                assert (len(ngh_idx) <= num_neighbors)
                assert (len(ngh_ts) <= num_neighbors)
                assert (len(ngh_eidx) <= num_neighbors)

                out_ngh_node_batch[i, num_neighbors - len(ngh_idx):] = ngh_idx
                out_ngh_t_batch[i, num_neighbors - len(ngh_ts):] = ngh_ts
                out_ngh_eidx_batch[i,
                                   num_neighbors - len(ngh_eidx):] = ngh_eidx

    return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch


# @jit
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


# @jit
def batch_interaction2subgraph(ngh_node_batch, ngh_eidx_batch):
    batch_size, num_neigh = ngh_node_batch.shape
    batch_n2n = np.zeros((batch_size, num_neigh, num_neigh))
    # batch_nids = np.zeros((batch_size, num_neigh), dtype=np.int32)
    batch_nids = np.zeros_like(ngh_node_batch)
    batch_e2n = np.zeros((batch_size, num_neigh, num_neigh))
    # batch_eids = np.zeros((batch_size, num_neigh), dtype=np.int32)
    batch_eids = np.zeros_like(ngh_eidx_batch)

    # for i, (ngh_node,
            # ngh_eidx) in enumerate(zip(ngh_node_batch, ngh_eidx_batch)):
    for i in range(len(ngh_node_batch)):
        ngh_node = ngh_node_batch[i]
        ngh_eidx = ngh_eidx_batch[i]
        n2n, nids, e2n, eids = sequence2graph(ngh_node, ngh_eidx)
        batch_n2n[i] = n2n
        batch_nids[i][:len(nids)] = nids
        batch_e2n[i] = e2n
        batch_eids[i] = eids
    return batch_n2n, batch_nids, batch_e2n, batch_eids


class SubgraphNeighborFinder:
    _dgl_path = "subgraph_cache/{task}-{dataset}-{m}.dgl"
    _np_path = "subgraph_cache/{task}-{dataset}-{m}.npz"
    PRECEISION = 5

    def __init__(self,
                 adj_list,
                 ts_l,
                 graph_type="numpy",
                 task="edge",
                 dataset="ia-contact",
                 uniform=False):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        self.ts_l = ts_l
        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(
            adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        assert (graph_type in ["numpy", "dgl"])
        self.type = graph_type
        self.task = task
        self.dataset = dataset
        self._ngh_cache = {}
        self._off_cache = {}
        self.uniform = uniform

        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def init_off_set(self, adj_list):
        """
        Params
        ------
        adj_list: List[List[int]]
        
        """
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

        assert (len(n_idx_l) == len(n_ts_l))
        assert (off_set_l[-1] == len(n_ts_l))

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def preprocess(self, num_neighbors=20):
        dgl_type = self.type == "dgl"

        start = time.time()
        if self.type == "numpy":
            mat_n2n, mat_nids, mat_e2n, mat_eids = subgraph_np(
                self.node_idx_l, self.node_ts_l, self.edge_idx_l,
                self.off_set_l, num_neighbors)
            mat_arr = {}
            mat_arr["mat_n2n"] = mat_n2n
            mat_arr["mat_nids"] = mat_nids
            mat_arr["mat_e2n"] = mat_e2n
            mat_arr["mat_eids"] = mat_eids

            self._ngh_cache[num_neighbors] = mat_arr
        elif self.type == "dgl":
            path = self._dgl_path.format(task=self.task,
                                         dataset=self.dataset,
                                         m=num_neighbors)
            if not os.path.exists(path):
                self.logger.warning("Neighbor cache %s not exists.", path)
                interaction2subgraph(self.dataset, num_neighbors, self.task,
                                     dgl_type)
            sg, _ = dgl.load_graphs(path)
            self._ngh_cache[num_neighbors] = sg
        else:
            raise NotImplementedError(self.type)

        end = time.time()
        self.logger.warning("Loading precomputation cache cost %.2fs.",
                            end - start)

    def find_before(self, src_idx, cut_time):
        """
    
        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        return find_before_nb(src_idx, cut_time, off_set_l, node_idx_l,
                              node_ts_l, edge_idx_l)

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        return get_temporal_neighbor_nb(src_idx_l, cut_time_l, num_neighbors,
                                        off_set_l, node_idx_l, node_ts_l,
                                        edge_idx_l, self.uniform)
    
    def batch_interaction2subgraph(self, src_idx_l, cut_time_l, num_neighbors=20):
        ngh_node_batch, ngh_eidx_batch, ngh_t_batch = self.get_temporal_neighbor(
            src_idx_l, cut_time_l, num_neighbors=num_neighbors)

        ngh_node_batch, ngh_eidx_batch
        batch_size, num_neigh = ngh_node_batch.shape
        batch_n2n = np.zeros((batch_size, num_neigh, num_neigh))
        # batch_nids = np.zeros((batch_size, num_neigh), dtype=np.int32)
        batch_nids = np.zeros_like(ngh_node_batch)
        batch_e2n = np.zeros((batch_size, num_neigh, num_neigh))
        # batch_eids = np.zeros((batch_size, num_neigh), dtype=np.int32)
        batch_eids = np.zeros_like(ngh_eidx_batch)

        for i in range(len(ngh_node_batch)):
            src_node, src_t = src_idx_l[i], cut_time_l[i]
            src_index = self.find_before_index(src_node, src_t)
            node_ts_key = self.make_key(src_node, src_index)
            if  node_ts_key in self._ngh_cache:
                n2n, nids, e2n, eids = self._ngh_cache[node_ts_key]
            else:
                ngh_node = ngh_node_batch[i]
                ngh_eidx = ngh_eidx_batch[i]
                n2n, nids, e2n, eids = sequence2graph(ngh_node, ngh_eidx)
                self._ngh_cache[node_ts_key] = (n2n, nids, e2n, eids)
            batch_n2n[i] = n2n
            batch_nids[i][:len(nids)] = nids
            batch_e2n[i] = e2n
            batch_eids[i] = eids
        
        batch_subgraph = (batch_n2n, batch_nids, batch_e2n, batch_eids)
        return ngh_t_batch, batch_subgraph

    def find_before_index(self, src_idx, cut_time):
        """Find the index of the latest interaction.

        Params
        ------
        src_idx: int
        cut_time: float
        """
        ans = self.check_cache(src_idx, cut_time)
        if ans is not None:
            return ans

        node_ts_l = self.node_ts_l
        off_set_l = self.off_set_l
        start, end = off_set_l[src_idx], off_set_l[src_idx + 1]

        neighbors_ts = node_ts_l[start:end]

        right = np.searchsorted(neighbors_ts, cut_time, side="left")

        if right - 1 >= 0:
            index = start + right - 1
        else:
            index = 0

        self.update_cache(src_idx, cut_time, index)

        return index

    def get_neighbor_np(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int

        Returns:
        ----------
        batch_n2n: (B, K, K)
        batch_nid: (B, K)
        batch_e2n: (B, K, M)
        batch_eid: (B, M)
        batch_ets: (B, M)
        """
        assert (self.type == "numpy")

        if num_neighbors not in self._ngh_cache:
            self.preprocess(num_neighbors)

        ngh_cache = self._ngh_cache[num_neighbors]
        mat_n2n = ngh_cache["mat_n2n"]  # (E, K, K)
        mat_nids = ngh_cache["mat_nids"]  # (E, K)
        mat_e2n = ngh_cache["mat_e2n"]  # (E, K, M)
        mat_eids = ngh_cache["mat_eids"]  # (E, M)

        node_ts_l = self.node_ts_l

        batch_n2n_l = []
        batch_nids_l = []
        batch_e2n_l = []
        batch_eids_l = []
        batch_ets_l = []

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            # find the index of the latest interaction
            left = self.find_before_index(src_idx, cut_time)
            n2n, nids, e2n, eids = mat_n2n[left], mat_nids[left], mat_e2n[
                left], mat_eids[left]
            ets = self.ts_l[eids]
            batch_n2n_l.append(n2n)
            batch_nids_l.append(nids)
            batch_e2n_l.append(e2n)
            batch_eids_l.append(eids)
            batch_ets_l.append(ets)

        batch_n2n = np.stack(batch_n2n_l, axis=0)
        batch_nids = np.stack(batch_nids_l, axis=0)
        batch_e2n = np.stack(batch_e2n_l, axis=0)
        batch_eids = np.stack(batch_eids_l, axis=0)
        batch_ets = np.stack(batch_ets_l, axis=0)
        return batch_n2n, batch_nids, batch_e2n, batch_eids, batch_ets

    def get_neighbors_dgl(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int

        Returns:
        ----------
        graphs: List[dgl.DGLGraph], sotring NID, EID, and timestamp.
        """
        assert (self.type == "dgl")

        if num_neighbors not in self._ngh_cache:
            self.preprocess(num_neighbors)

        ngh_graphs = self._ngh_cache[num_neighbors]
        node_ts_l = self.node_ts_l
        batch_graphs = []
        batch_nids = []
        batch_ts = []

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            # find the index of the latest interaction
            left = self.find_before(src_idx, cut_time)
            sg = ngh_graphs[left]
            sg_eid = sg.edata[dgl.EID].numpy()
            sg.edata["timestamp"] = torch.tensor(node_ts_l[sg_eid]).float()
            batch_graphs.append(sg)
            batch_nids.append(sg.ndata[dgl.NID].numpy())
            batch_ts.append(np.repeat(cut_time, sg.number_of_nodes()))

        return batch_graphs, batch_nids, batch_ts

    def make_key(self, node, index):
        key = "{}-{}".format(node, index)
        return key

    def update_cache(self, node, ts, results):
        key = (node, round(ts, self.PRECEISION))
        if key not in self._off_cache:
            self._off_cache[key] = results

    def check_cache(self, node, ts):
        key = (node, round(ts, self.PRECEISION))
        return self._off_cache.get(key)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Helper for preprocessing interaction subgraphs.")
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    parser.add_argument("-d", "--data", default="fb-forum", type=str)
    parser.add_argument("-gt",
                        "--graph-type",
                        default="numpy",
                        choices=["dgl", "numpy"])
    parser.add_argument("-n", "--num-neighbors", default=20, type=int)
    args = parser.parse_args()
    logger.info(args)

    TASK = args.task
    DATA = args.data
    GRAPH_TYPE = args.graph_type
    NUM_NGH = args.num_neighbors

    edges = load_data_var(DATA, TASK)
    adj_list = init_adj(edges)
    ts_l = edges["timestamp"].to_numpy()
    ngh_finder = SubgraphNeighborFinder(adj_list, ts_l, GRAPH_TYPE, TASK, DATA)

    BATCHSIZE = 200

    src_l = edges["from_node_id"].to_numpy()
    dst_l = edges["to_node_id"].to_numpy()
    ts_l = edges["timestamp"].to_numpy()

    num_batch = int(math.ceil(len(src_l) / BATCHSIZE))
    for k in trange(num_batch):
        s_idx = k * BATCHSIZE
        e_idx = min(len(src_l), s_idx + BATCHSIZE)
        src_l_cut, dst_l_cut, ts_l_cut = src_l[s_idx:e_idx], dst_l[
            s_idx:e_idx], ts_l[s_idx:e_idx]
        if GRAPH_TYPE == "numpy":
            _, nids, _, _, ts = ngh_finder.get_neighbor_np(
                src_l_cut, ts_l_cut, NUM_NGH)
            _ = ngh_finder.get_neighbor_np(nids.flatten(), ts.flatten(),
                                           NUM_NGH)
            _, nids, _, _, ts = ngh_finder.get_neighbor_np(
                dst_l_cut, ts_l_cut, NUM_NGH)
            _ = ngh_finder.get_neighbor_np(nids.flatten(), ts.flatten(),
                                           NUM_NGH)
        elif GRAPH_TYPE == "dgl":
            batch_graphs, batch_nids, batch_ts = ngh_finder.get_neighbors_dgl(
                src_l_cut, ts_l_cut, NUM_NGH)
            nids = np.concatenate(batch_nids)
            ts = np.concatenate(batch_ts)
            _ = ngh_finder.get_neighbors_dgl(nids, ts)
            batch_graphs, batch_nids, batch_ts = ngh_finder.get_neighbors_dgl(
                dst_l_cut, ts_l_cut, NUM_NGH)
            nids = np.concatenate(batch_nids)
            ts = np.concatenate(batch_ts)
            _ = ngh_finder.get_neighbors_dgl(nids, ts)
        else:
            raise NotImplementedError(GRAPH_TYPE)
