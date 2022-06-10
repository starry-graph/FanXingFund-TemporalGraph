from asyncio.log import logger
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm, trange
import easydict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
from model import compute_loss

from hdfs.client import Client
import json

from batch_model import BatchModel
from util import set_logger
from build_data import get_data
from IPython import embed



def train_model(args, model, train_loader, features, opt):
    for epoch in range(args.epochs):
        loss_avg, y_probs, y_labels = 0, [], []
        model.train()
        batch_bar = tqdm(train_loader, desc='train')
        for step, (input_nodes, pos_graph, neg_graph, history_blocks) in enumerate(batch_bar):
            history_inputs = [nfeat[nodes].to(args.device) for nfeat, nodes in zip(features, input_nodes)]
            # batch_inputs = nfeats[input_nodes].to(device)
            pos_graph = pos_graph.to(args.device)
            neg_graph = neg_graph.to(args.device)
            # blocks = [block.int().to(device) for block in blocks]
            history_blocks = [[block.int().to(args.device) for block in blocks] for blocks in history_blocks]

            pos_score, neg_score = model(history_blocks, history_inputs, pos_graph, neg_graph)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_avg += loss.item()

            y_probs.append(pos_score.detach().cpu().numpy())
            y_labels.append(np.ones_like(y_probs[-1]))
            y_probs.append(neg_score.detach().cpu().numpy())
            y_labels.append(np.zeros_like(y_probs[-1]))

            batch_bar.set_postfix(loss=round(loss.item(), 4))
        loss_avg /= len(train_loader)
        y_prob = np.hstack([y.squeeze(1) for y in y_probs])
        y_pred = y_prob > 0.5
        y_label = np.hstack([y.squeeze(1) for y in y_labels])

        acc = accuracy_score(y_label, y_pred)
        ap = average_precision_score(y_label, y_prob)
        auc = roc_auc_score(y_label, y_prob)
        f1 = f1_score(y_label, y_pred)
        logger.info('Epoch %03d Training loss: %.4f, ACC: %.4f, F1: %.4f, AP: %.4f, AUC: %.4f', \
            epoch, loss_avg, acc, f1, ap, auc)

    df = pd.DataFrame({'Loss': [loss_avg], 'ACC': [acc], 'F1': [f1], 'AP': [ap] ,'AUC': [auc]})
    df.to_csv(args.rst_path, index=False)

    logger.info(f'Saving model at {args.model_path}')
    torch.save(model.state_dict(), args.model_path)


def run(args):
    logger.info(f'Loading dataset {args.dataset} from {args.root_dir}')
    train_loader, features, n_features, num_nodes, num_edges = get_data(args, logger, mode='train')

    model = BatchModel(n_features, args.n_hidden, args.embed_dim, args.n_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters())
    
    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    train_model(args, model, train_loader, features, opt)


def config2args(config, args):
    args.dataset = config['spaceId']
    args.outfile_path = config['outFilePath']
    args.model_path = config['modelPath']
    args.feature_names = config['featureNames']
    txt = config['flinkFeatureNames']
    args.named_feats = [ord(s.lower())-ord('a') for s in txt if ord('A') <= ord(s) <=ord('z')] if txt!='all' else 'all'
    args.timespan_start = int(config['startTime'])
    args.timespan_end = int(config['endTime'])
    args.root_dir = config['dataPath']
    return args


def train(config):
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
        'cpp_file': "./wart-servers/examples/test.wasm"
    })

    logger = set_logger()
    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    args = config2args(config, args)
    logger.info(args)
    
    args.rst_path = os.path.join(args.outfile_path, 'train_metrics.csv')

    run(args)
    return args.outfile_path


def get_config():
    client = Client("http://192.168.1.13:9009")
    lines = []
    with client.read('/sxx/conf.json') as reader:
        for line in reader:
            lines.append(line)
    lines_utf8 = [line.decode() for line in lines]
    lines_replace = [line.replace('\xa0', '') for line in lines_utf8]
    config = json.loads(''.join(lines_replace))
    return config


if __name__ == '__main__':
    # config = {
    #     "taskId": "585838793082061314TSN",
    #     "spaceId": "dblp-coauthors",
    #     "outFilePath": "./results/",
    #     "modelPath": "./saved_models/dblp-coauthors_2epochs.pth",
    #     "featureNames":"属性A,属性B,属性C",
    #     "flinkFeatureNames":"属性A,属性D,属性E",
    #     "startTime": "2000",
    #     "endTime": "2020",
    #     "trainTarget": 1,
    #     "dataPath": "./",
    #     "otherParam": "",
    #     "labelName": "1",
    #     "idIndex": "1"
    # }
    config = get_config()
    outfile_path = train(config)
    print('outfile_path: ', outfile_path)