import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch

# NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/notebook/'
NOTE_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/Academic_GNN_Module/'
DBLP_PATH = NOTE_PATH + 'dblpv13/'
CSRA_PATH = '/nfs3-p1/zty/2021-FanXing/Academic-Networks/CSrankings/'

def timestamp_transform(config: Dict[str, str], args: argparse.ArgumentParser, logger):
    tstart = float(config['startTime'])
    tend = float(config['endTime'])
    # Processing the config's timespan for each dataset.
    if args.dataset == 'DBLPV13':
        # We get the year here, where tstart is at milliesecond.
        logger.warning('Set the timespan for DBLPV13.')
        tstart = datetime.fromtimestamp(tstart / 1e3).year
        tend = datetime.fromtimestamp(tend / 1e3).year
    elif args.dataset == 'ia_contact':
        # We compute the offset seconds from (2000, 1, 1).
        logger.warning('Set the timespan for ia_contact.')
        offset = datetime(2000, 1, 1).timestamp() * 1e3
        tstart = (tstart - offset) / 1e3
        tend = (tend - offset) / 1e3
    else:
        logger.warning('Use the config timespan for %s', args.dataset)

    return tstart, tend


def set_logger():
    # set up logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    return logger


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_result(val_metrics,
                 metrics,
                 dataset,
                 params,
                 postfix="GTC",
                 results="results"):
    res_path = "{}/{}-{}.csv".format(results, dataset, postfix)
    val_keys = val_metrics.keys()
    test_keys = metrics.keys()
    headers = ["method", "dataset"
               ] + list(val_keys) + list(test_keys) + ["params"]
    if not os.path.exists(res_path):
        f = open(res_path, 'w')
        f.write(",".join(headers) + "\r\n")
        f.close()
        os.chmod(res_path, 0o777)
    with open(res_path, 'a') as f:
        result_str = "{},{}".format(postfix, dataset)
        result_str += "," + ",".join(
            ["{:.4f}".format(val_metrics[k]) for k in val_keys])
        result_str += "," + ",".join(
            ["{:.4f}".format(metrics[k]) for k in test_keys])
        logging.info(result_str)
        params_str = ",".join(
            ["{}={}".format(k, v) for k, v in params.items()])
        params_str = "\"{}\"".format(params_str)
        row = result_str + "," + params_str + "\r\n"
        f.write(row)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):

        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(
                self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


def get_free_gpu():
    import gpustat
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(
        lambda gpu: float(gpu.entry['memory.total']) - float(gpu.entry[
            'memory.used']), stats)
    pairs = list(zip(ids, ratios))
    random.shuffle(pairs)
    bestGPU = max(pairs, key=lambda x: x[1])[0]
    print("setGPU: Setting GPU to: {}".format(bestGPU))
    return str(bestGPU)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print("%r  %2.2f s" % (method.__name__, te - ts))
        return result

    return timed
