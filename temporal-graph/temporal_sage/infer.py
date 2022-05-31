from asyncio.log import logger
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
import easydict

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
from batch_model import BatchModel
from model import Model, compute_loss, construct_negative_graph
from util import (CSRA_PATH, DBLP_PATH, NOTE_PATH, EarlyStopMonitor,
                  set_logger, set_random_seed, write_result)
from build_data import get_data
from IPython import embed


def infer_model(args, model, test_loader, features):
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    y_probs, y_labels = [], []
    from_nodes, to_nodes = [], []
    with torch.no_grad():
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(tqdm(test_loader, desc='infer')):
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

            nids =  pos_graph.ndata['_ID']
            from_node, to_node = pos_graph.edges()
            from_nodes.append(nids[from_node].detach().cpu().numpy())
            to_nodes.append(nids[to_node].detach().cpu().numpy())

            nids =  neg_graph.ndata['_ID']
            from_node, to_node = neg_graph.edges()
            from_nodes.append(nids[from_node].detach().cpu().numpy())
            to_nodes.append(nids[to_node].detach().cpu().numpy())

    y_prob = np.hstack([y.squeeze(1) for y in y_probs])
    y_pred = y_prob > 0.5
    y_label = np.hstack([y.squeeze(1) for y in y_labels])

    acc = accuracy_score(y_label, y_pred)
    ap = average_precision_score(y_label, y_prob)
    auc = roc_auc_score(y_label, y_prob)
    f1 = f1_score(y_label, y_pred)

    logger.info('Test ACC: %.4f, F1: %.4f, AP: %.4f, AUC: %.4f', acc, f1, ap, auc)
    df = pd.DataFrame({'ACC': [acc], 'F1': [f1], 'AP': [ap] ,'AUC': [auc]})
    df.to_csv(args.rst_path, index=False)

    from_nid = np.hstack([y for y in from_nodes])
    to_nid = np.hstack([y for y in to_nodes])
    df = pd.DataFrame({'from_nid': from_nid, 'to_nid': to_nid, 'label': y_label ,'prob': y_prob})
    df.to_csv(args.pred_path, index=False)
    
    return args.outfile_path


def run(args):
    logger.info(f'Loading dataset {args.dataset} from {args.root_dir}')
    test_loader, features, n_features, num_nodes, num_edges = get_data(args, logger)

    model = BatchModel(n_features, args.n_hidden, args.embed_dim, args.n_layers).to(args.device)
    logger.info(f'Loading model from {args.model_path}')
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    
    logger.info('Begin infering with %d nodes, %d edges.', num_nodes, num_edges)
    infer_model(args, model, test_loader, features)


def config2args(config, args):
    args.dataset = config['spaceId']
    args.outfile_path = config['outFilePath']
    args.model_path = config['modelPath']
    args.feature_names = config['featureNames']
    txt = config['flinkFeatureNames']
    args.named_feats = [ord(s.lower())-ord('a') for s in txt if ord('A') <= ord(s) <=ord('z')]
    args.timespan_start = int(config['startTime'])
    args.timespan_end = int(config['endTime'])
    args.root_dir = config['dataPath']
    return args


def infer(config):
    args = easydict.EasyDict({
        'dataset': 'ia-contact', 
        'root_dir': './', 
        'prefix': 'TemporalSAGE', 
        'epochs': 50, 
        'bs': 1024, 
        'num_ts': 20,
        'n_hidden': 100, 
        'embed_dim': 100, 
        'n_layers': 2, 
        'gpu': 0, 
        'lr': 1e-2, 
        'temporal_feat': False, 
        'named_feats': 'all', 
        'timespan_start': -np.inf, 
        'timespan_end': np.inf, 
    })

    logger = set_logger()
    args.device = torch.device('cuda:{}'.format(args.gpu))
    args = config2args(config, args)
    logger.info(args)

    PARAM_STR = f'{args.epochs}-{args.bs}-{args.num_ts}-{args.n_hidden}'
    PARAM_STR += f'-{args.embed_dim}-{args.n_layers}-{args.lr}'
    PARAM_STR += f'-{args.temporal_feat}-{args.named_feats}'
    SAVE_PATH = f'{args.prefix}-{PARAM_STR}-{args.timespan_start}-{args.timespan_end}-{args.dataset}'
    
    args.pred_path = os.path.join(args.outfile_path, SAVE_PATH + '.prediction')
    args.rst_path = os.path.join(args.outfile_path, SAVE_PATH + '.result')

    run(args)
    return args.outfile_path


if __name__ == '__main__':
    config = {
        "taskId": "585838793082061314TSN",
        "spaceId": "fb-forum",
        "outFilePath": "./saved_models/",
        "modelPath": "./saved_models/temp.pth",
        "featureNames":"属性A,属性B,属性C",
        "flinkFeatureNames":"属性A,属性D,属性E",
        "startTime": "1095290000",
        "endTime": "1096500000",
        "trainTarget": 1,
        "dataPath": "./",
        "otherParam": "",
        "labelName": "1",
        "idIndex": "1"
    }

    outfile_path = infer(config)
    print('outfile_path: ', outfile_path)