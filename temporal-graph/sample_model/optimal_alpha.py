import argparse
import logging
import resource
import time
from collections import defaultdict

import cvxpy as cp
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.lin_ops.lin_utils import multiply
import numpy as np
import torch
from joblib import Parallel, delayed
from numba import jit
from tqdm import trange

from data_loader.data_util import _iterate_datasets
from data_loader.data_util import load_graph, load_data
from sample_model.graph import NeighborFinder


def optimal_alpha(offset_l, node_idx_l, node_ts_l, latest=True):
    ''' We use cvxpy for optimization of the log maximum likelihood of link
        prediction based multinomial sampling probability estimation.
        `argmax \Pi_{ij} \\frac{e^{\\alpha * t_j}}{ \sum_k e^{\\alpha * t_k}}`
    =>  `argmin \sum_{ij} \log {sum_k e^{\\alpha * t_k} - \\alpha * t_j}`
    '''

    start = time.time()

    def _solve_alpha(k):
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (10 * (2**30), hard))

        ALPHA = cp.Variable()
        left = offset_l[k]
        right = offset_l[k + 1]
        loss = 0.
        if right - left > 100:
            term_range = np.random.randint(left, right, 100)
        else:
            term_range = np.arange(left, right)
        for i in term_range:
            cur_idx = node_idx_l[left:i]
            cur_ts = node_ts_l[left:i]
            indices = np.where(cur_idx == node_idx_l[i])
            if len(indices[0]) <= 0:
                continue

            if latest:
                indices = indices[0][-1]
            loss += cp.log_sum_exp(cp.multiply(ALPHA, cur_ts)) \
                - ALPHA * cur_ts[indices]
        obj = cp.Minimize(loss)
        constraint = [0 <= ALPHA, ALPHA <= 100]
        problem = cp.Problem(obj, constraint)

        try:
            result = problem.solve(solver="ECOS", max_iters=50)
            a_ = ALPHA.value
        except Exception as e:
            a_ = ALPHA.value
        if a_ is None:
            return 100.0
        else:
            return a_

    n_node = len(offset_l) - 1
    result = Parallel(n_jobs=30, verbose=10)(delayed(_solve_alpha)(k)
                                             for k in range(n_node))
    end = time.time()
    logging.info("Optimal alpha construction cost %.2f seconds.", end - start)
    return np.array(result)


def optimal_alpha_torch(offset_l, node_idx_l, node_ts_k, latest=True):
    ALPHA = torch.zeros(1, requires_grad=True)
    pass


def prepare_data(dataset=None, task="edge", train_ratio=0.7):
    logging.info("Begin dataset %s.", dataset)
    if task == "edge":
        edges, n_nodes, val_time, test_time = load_graph(dataset=dataset)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(g_df) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]

        train_df = g_df[g_df["ts"] < val_time]

    elif task == "node":
        edges, nodes = load_data(dataset, "format")
        # padding node is 0, so add 1 here.
        id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
        edges["from_node_id"] = edges["from_node_id"].map(id2idx)
        edges["to_node_id"] = edges["to_node_id"].map(id2idx)
        g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
        g_df["idx"] = np.arange(1, len(edges) + 1)
        g_df.columns = ["u", "i", "ts", "idx"]

        val_time = np.quantile(g_df.ts, train_ratio)
        train_df = g_df[g_df["ts"] < val_time]

    src_l = train_df.u.values
    dst_l = train_df.i.values
    e_idx_l = train_df.idx.values
    ts_l = train_df.ts.values
    max_idx = max(g_df["u"].max(), g_df["i"].max())
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    ngh_finder = NeighborFinder(adj_list)
    return ngh_finder.off_set_l, ngh_finder.node_idx_l, ngh_finder.norm_ts_l


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute optimal alpha for multinomial distribution.")
    parser.add_argument("-t",
                        "--task",
                        default="edge",
                        choices=["edge", "node"])
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # JODIE datasets for node classification
    if args.task == "edge":
        datasets = _iterate_datasets()
        for data in datasets:
            offset_l, node_idx_l, norm_ts_l = prepare_data(data, "edge")
            alpha = optimal_alpha(offset_l, node_idx_l, norm_ts_l, latest=True)
            np.savez(f"sample_cache/{args.task}-{data}-alpha", alpha=alpha)
    elif args.task == "node":
        datasets = ["JODIE-wikipedia", "JODIE-mooc", "JODIE-reddit"]
        for data in datasets:
            offset_l, node_idx_l, norm_ts_l = prepare_data(data,
                                                           "node",
                                                           train_ratio=0.7)
            alpha = optimal_alpha(offset_l, node_idx_l, norm_ts_l, latest=True)
            np.savez(f"sample_cache/{args.task}-{data}-alpha", alpha=alpha)
    else:
        raise NotImplementedError(args.task)
