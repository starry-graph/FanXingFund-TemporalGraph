import logging
import numpy as np
import torch
from numba import jit, prange
from joblib import Parallel, delayed

def make_label_data(src_l, dst_l, ts_l, val_flag, rand_sampler):
    num = np.sum(val_flag)
    val_src = src_l[val_flag]
    val_dst = dst_l[val_flag]
    val_ts = ts_l[val_flag]
    val_src_l = np.hstack([val_src, val_src])
    _, dst_fake = rand_sampler.sample(num)
    val_dst_l = np.hstack([val_dst, dst_fake])
    val_ts_l = np.hstack([val_ts, val_ts])
    val_label_l = np.hstack([np.ones(num), np.zeros(num)])
    return val_src_l, val_dst_l, val_ts_l, val_label_l

@jit
def find_before_nb(src_idx, cut_time, off_set_l, node_idx_l, node_ts_l,
                   edge_idx_l):
    neighbors_idx = node_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_ts = node_ts_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]
    neighbors_e_idx = edge_idx_l[off_set_l[src_idx]:off_set_l[src_idx + 1]]

    if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
        return neighbors_idx, neighbors_ts, neighbors_e_idx

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

class NeighborFinder:
    PRECISION = 5

    def __init__(self, adj_list, uniform=False, exp=False, alpha=1.0):
        """
        Params
        ------
        node_idx_l: List[int]
        node_ts_l: List[int]
        off_set_l: List[int], such that node_idx_l[off_set_l[i]:off_set_l[i + 1]] = adjacent_list[i]
        """

        node_idx_l, node_ts_l, edge_idx_l, off_set_l = self.init_off_set(adj_list)
        self.node_idx_l = node_idx_l
        self.node_ts_l = node_ts_l
        dt = node_ts_l.max() - node_ts_l.min()
        self.norm_ts_l = (node_ts_l - node_ts_l.min()) / dt
        self.edge_idx_l = edge_idx_l

        self.off_set_l = off_set_l

        # Backward capability.
        assert not(uniform and exp)
        self.uniform = uniform
        if uniform:
            self.sampling = "uniform"
        elif exp:
            self.sampling = "exp"
        else:
            self.sampling = "temporal"

        if type(alpha) is float:
            self.alpha = np.full_like(off_set_l, alpha)
        elif type(alpha) is not np.ndarray:
            self.alpha = np.zeros_like(off_set_l)
        else:
            self.alpha = alpha
        
        self.cache = {}

        numba_logger = logging.getLogger("numba")
        numba_logger.setLevel(logging.WARNING)

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

        assert len(n_idx_l) == len(n_ts_l)
        assert off_set_l[-1] == len(n_ts_l)

        return n_idx_l, n_ts_l, e_idx_l, off_set_l

    def find_before(self, src_idx, cut_time, norm=False):
        """

        Params
        ------
        src_idx: int
        cut_time: float
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        norm_ts_l = self.norm_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l

        neighbors_idx = node_idx_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        neighbors_ts = node_ts_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        neighbors_norm_ts = norm_ts_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]
        neighbors_e_idx = edge_idx_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]

        if len(neighbors_idx) == 0 or len(neighbors_ts) == 0:
            if norm:
                return neighbors_idx, neighbors_e_idx, neighbors_ts, neighbors_norm_ts
            else:
                return neighbors_idx, neighbors_e_idx, neighbors_ts

        right = np.searchsorted(neighbors_ts, cut_time, side="left")
        ngh_idx = neighbors_idx[:right]
        ngh_eidx = neighbors_e_idx[:right]
        ngh_ts = neighbors_ts[:right]
        norm_ts = neighbors_norm_ts[:right]

        if norm:
            return ngh_idx, ngh_eidx, ngh_ts, norm_ts
        else:
            return ngh_idx, ngh_eidx, ngh_ts
    
    def find_before_idx(self, src_idx, cut_time, norm=False):
        node_ts_l = self.node_ts_l
        off_set_l = self.off_set_l
        neighbors_ts = node_ts_l[off_set_l[src_idx] : off_set_l[src_idx + 1]]

        if len(neighbors_ts) == 0 or len(neighbors_ts) == 0:
            return 0

        right = np.searchsorted(neighbors_ts, cut_time, side="left")
        return right

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        """
        assert len(src_idx_l) == len(cut_time_l)

        if self.sampling == "exp":
            return self.exp_sampling(src_idx_l, cut_time_l, num_neighbors)

        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        return get_temporal_neighbor_nb(src_idx_l, cut_time_l, num_neighbors, off_set_l, node_idx_l, node_ts_l, edge_idx_l, self.uniform)

    def exp_sampling(self, src_idx_l, cut_time_l, num_neighbors=20):
        out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)
        out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.float32)
        out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbors)).astype(np.int32)

        for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            right = self.find_before_idx(src_idx, cut_time)
            result = self.check_cache(src_idx, right)
            if result is not None:
                out_ngh_node_batch[i] = result[0]
                out_ngh_t_batch[i] = result[1]
                out_ngh_eidx_batch[i] = result[2]
                continue

            ngh_idx, ngh_eidx, ngh_ts, norm_ts = self.find_before(
                src_idx, cut_time, norm=True
            )
            if len(ngh_idx) <= 0:
                continue

            if len(ngh_idx) < num_neighbors:
                right = len(ngh_idx)
                out_ngh_node_batch[i, :right] = ngh_idx
                out_ngh_t_batch[i, :right] = ngh_ts
                out_ngh_eidx_batch[i, :right] = ngh_eidx
                continue

            ngh_dt = norm_ts - np.max(norm_ts)
            ngh_logit = np.exp(self.alpha[src_idx] * ngh_dt)
            prob = ngh_logit / np.sum(ngh_logit)
            nonzero_num = (prob > 0).sum()
            num = min(num_neighbors, nonzero_num)
            sampled_idx = np.random.choice(len(ngh_ts), size=num, replace=False, p=prob)
            sampled_idx = np.sort(sampled_idx)

            out_ngh_node_batch[i, :num] = ngh_idx[sampled_idx]
            out_ngh_t_batch[i, :num] = ngh_ts[sampled_idx]
            out_ngh_eidx_batch[i, :num] = ngh_eidx[sampled_idx]

            # resort based on time
            pos = out_ngh_t_batch[i, :].argsort()
            out_ngh_node_batch[i, :] = out_ngh_node_batch[i, :][pos]
            out_ngh_t_batch[i, :] = out_ngh_t_batch[i, :][pos]
            out_ngh_eidx_batch[i, :] = out_ngh_eidx_batch[i, :][pos]

            result = (out_ngh_node_batch[i], out_ngh_t_batch[i], out_ngh_eidx_batch[i])
            self.update_cache(src_idx, right, result)            


        return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch

    def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors=20):
        """Sampling the k-hop sub graph"""
        x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors)
        node_records = [x]
        eidx_records = [y]
        t_records = [z]
        for _ in range(k - 1):
            ngn_node_est, ngh_t_est = (
                node_records[-1],
                t_records[-1],
            )  # [N, *([num_neighbors] * (k - 1))]
            orig_shape = ngn_node_est.shape
            ngn_node_est = ngn_node_est.flatten()
            ngn_t_est = ngh_t_est.flatten()
            (
                out_ngh_node_batch,
                out_ngh_eidx_batch,
                out_ngh_t_batch,
            ) = self.get_temporal_neighbor(ngn_node_est, ngn_t_est, num_neighbors)
            out_ngh_node_batch = out_ngh_node_batch.reshape(
                *orig_shape, num_neighbors
            )  # [N, *([num_neighbors] * k)]
            out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(*orig_shape, num_neighbors)
            out_ngh_t_batch = out_ngh_t_batch.reshape(*orig_shape, num_neighbors)

            node_records.append(out_ngh_node_batch)
            eidx_records.append(out_ngh_eidx_batch)
            t_records.append(out_ngh_t_batch)
        return node_records, eidx_records, t_records

    def update_cache(self, node, idx, results):
        key = (node, idx)
        if key not in self.cache:
            self.cache[key] = results

    def check_cache(self, node, idx):
        key = (node, idx)
        return self.cache.get(key)
