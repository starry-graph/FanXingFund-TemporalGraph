from asyncio.log import logger
import os
import numpy as np
import pandas as pd
import time
import torch
# from tqdm import tqdm, trange
import easydict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score

from hdfs.client import Client
import json
import urllib.parse as urlparse
import pickle as pkl
import argparse

from batch_model import BatchModel
from util import set_logger
from build_data import get_data



def infer_model(args, model, test_loader, features):
    model.eval()
    y_probs, y_labels = [], []
    from_nodes, to_nodes, y_timespan = [], [], []
    test_start = time.time()
    with torch.no_grad():
        # for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(tqdm(test_loader, desc='infer')):
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(test_loader):
            batch_start = time.time()
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

            nids, cur_ts =  pos_graph.ndata['_ID'], pos_graph.edata['ts']
            from_node, to_node = pos_graph.edges()
            from_nodes.append(nids[from_node].detach().cpu().numpy())
            to_nodes.append(nids[to_node].detach().cpu().numpy())
            y_timespan.append(cur_ts.detach().cpu().numpy())

            from_node, to_node = neg_graph.edges()
            from_nodes.append(nids[from_node].detach().cpu().numpy())
            to_nodes.append(nids[to_node].detach().cpu().numpy())

            neg_num = len(from_node) // len(cur_ts)
            cur_ts = cur_ts.repeat_interleave(neg_num)
            y_timespan.append(cur_ts.detach().cpu().numpy())

            batch_time = time.time() - batch_start
            print('\r Current batch: {}/{} costs {:.2f} seconds.'.format(str(step).zfill(4), len(test_loader), batch_time), end='')

    y_prob = np.hstack([y.squeeze(1) for y in y_probs])
    y_pred = y_prob > 0.5
    y_label = np.hstack([y.squeeze(1) for y in y_labels])

    acc = accuracy_score(y_label, y_pred)
    ap = average_precision_score(y_label, y_prob)
    auc = roc_auc_score(y_label, y_prob)
    f1 = f1_score(y_label, y_pred)

    test_time = time.time() - test_start
    print('Test costs {:.2f} seconds.'.format(test_time))
    logger.info('Test ACC: %.4f, F1: %.4f, AP: %.4f, AUC: %.4f', acc, f1, ap, auc)
    df = pd.DataFrame({'ACC': [acc], 'F1': [f1], 'AP': [ap] ,'AUC': [auc]})
    with args.rst_client.write(args.rst_path, encoding='utf-8', overwrite=True) as writer:
        df.to_csv(writer, index=False)

    from_nid = np.hstack([y for y in from_nodes])
    to_nid = np.hstack([y for y in to_nodes])
    y_ts = np.hstack([y for y in y_timespan])
    df = pd.DataFrame({'from_nid': from_nid, 'to_nid': to_nid, 'label': y_label ,'prob': y_prob, 'timestamp': y_ts})
    with args.rst_client.write(args.pred_path, encoding='utf-8', overwrite=True) as writer:
        df.to_csv(writer, index=False)

    return args.outfile_path


def run(args):
    logger.info(f'Loading dataset {args.dataset} from {args.root_dir}')
    test_loader, features, n_features, num_nodes, num_edges = get_data(args, logger, mode='infer')

    model = BatchModel(n_features, args.n_hidden, args.embed_dim, args.n_layers).to(args.device)
    logger.info(f'Loading model from {args.model_path}')

    with args.model_client.read(args.model_path) as reader:
        model_dict = pkl.load(reader)
    model.load_state_dict(model_dict)
    
    logger.info('Begin infering with %d nodes, %d edges.', num_nodes, num_edges)
    infer_model(args, model, test_loader, features)


def config2args(config, args):
    args.dataset = config['spaceId']
    args.outfile_path = config['outFilePath']
    args.model_path = config['modelPath']
    args.feature_names = config['featureNames']
    txt = config['flinkFeatureNames']
    args.named_feats = 'all' # [ord(s.lower())-ord('a') for s in txt if ord('A') <= ord(s) <=ord('z')] if txt!='all' else 'all'
    args.timespan_start = int(config['startTime'])
    args.timespan_end = int(config['endTime'])
    args.dgl_sampler = config['dgl_sampler']
    # args.root_dir = config['dataPath']
    return args


def infer(config):
    args = easydict.EasyDict({
        'dataset': 'ia-contact', 
        'root_dir': './', 
        'prefix': 'TemporalSAGE', 
        'epochs': 50, 
        'bs': 1024, 
        'num_ts': 19,
        'n_hidden': 100, 
        'embed_dim': 100, 
        'n_layers': 2, 
        'gpu': 0, 
        'lr': 1e-2, 
        'temporal_feat': False, 
        'named_feats': 'all', 
        'timespan_start': -np.inf, 
        'timespan_end': np.inf, 
        'cpp_file': "./wart-servers/examples/sampler.wasm"
    })

    logger = set_logger()
    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    args = config2args(config, args)
    logger.info(args)

    # PARAM_STR = f'{args.epochs}-{args.bs}-{args.num_ts}-{args.n_hidden}'
    # PARAM_STR += f'-{args.embed_dim}-{args.n_layers}-{args.lr}'
    # PARAM_STR += f'-{args.temporal_feat}-{args.named_feats}'
    # SAVE_PATH = f'{args.prefix}-{PARAM_STR}-{args.timespan_start}-{args.timespan_end}-{args.dataset}'
    # SAVE_PATH = f'{args.dataset}'
    
    client_path, file_path = split_url(args.outfile_path)
    args.rst_client = Client(client_path)
    args.pred_path = os.path.join(file_path, 'prediction.csv')
    args.rst_path = os.path.join(file_path, 'infer_metrics.csv')

    client_path, args.model_path = split_url(args.model_path)
    args.model_client = Client(client_path)

    run(args)
    return args.outfile_path


def split_url(url):
    uparse = urlparse.urlparse(url)
    client_path = "http://" + uparse.netloc
    file_path = uparse.path
    return client_path, file_path


def get_config(url):
    client_path, config_path = split_url(url)

    client = Client(client_path)
    lines = []
    with client.read(config_path) as reader:
        for line in reader:
            lines.append(line)
    lines_utf8 = [line.decode() for line in lines]
    lines_replace = [line.replace('\xa0', '') for line in lines_utf8]
    config = json.loads(''.join(lines_replace))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default='http://192.168.1.13:9009/dev/conf/infer.json')
    parser.add_argument("--dgl_sampler", "-s", action='store_true')
    args = parser.parse_args()

    config = get_config(args.config)
    print(config)
    # config = {
    #     "taskId": "585838793082061314TSN",
    #     "spaceId": "DBLPV13",
    #     # "outFilePath": "./saved_models/",
    #     "outFilePath": "http://192.168.1.13:9009/dev/pytorch/infer-93B9AE168267/",
    #     # "modelPath": "./saved_models/dblp-coauthors_2epochs.pth",
    #     "modelPath": "http://192.168.1.13:9009/dev/pytorch/train-4FB003D56AAC/train-4FB003D56AAC.pth",
    #     "featureNames":"属性A,属性B,属性C",
    #     "flinkFeatureNames":"属性A,属性D,属性E",
    #     "startTime": "2001",
    #     "endTime": "2003",
    #     "trainTarget": 1,
    #     "dataPath": "./",
    #     "otherParam": "",
    #     "labelName": "1",
    #     "idIndex": "1"
    # }
    config['dgl_sampler'] = args.dgl_sampler

    outfile_path = infer(config)
    print('outfile_path: ', outfile_path)