from asyncio.log import logger
import os
import numpy as np
import pandas as pd
import time
import torch
from tqdm import tqdm, trange
import easydict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, average_precision_score
from model import compute_loss

from hdfs.client import Client
import json
import urllib.parse as urlparse
import pickle as pkl
import argparse

from batch_model import BatchModel
from util import set_logger
from build_data import get_data



def train_model(args, model, train_loader, features, opt):
    for epoch in range(args.epochs):
        loss_avg, y_probs, y_labels = 0, [], []
        model.train()
        # batch_bar = tqdm(train_loader, desc='train')
        batch_bar = train_loader
        epoch_start = time.time()
        batch_start = time.time()
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

            # batch_bar.set_postfix(loss=round(loss.item(), 4))
            batch_time = time.time() - batch_start
            batch_start = time.time()
            
            if args.dgl_sampler:
                sampler_str = 'Using Dgl Neighbor Sampler.'
            else:
                sampler = train_loader.sampler
                start_time = np.min(sampler.resp_start_times)
                end_time = np.max(sampler.resp_end_times)
                query_counts = np.sum(sampler.resp_query_counts)
                node_counts = np.sum(sampler.resp_node_counts)
                sampler.clear_resp_metrics()
                sampler_str = ' Sampler services costs {} milliseconds with {} nodes.'.format(end_time - start_time, node_counts) 
            batch_str = '\r Current batch: {}/{} costs {:.2f} seconds.'.format(str(step).zfill(4), len(batch_bar), batch_time)
            print(batch_str + sampler_str, end='')


        epoch_time = time.time() - epoch_start
        print('\n Epoch {:03d} costs {:.2f} seconds.'.format(epoch, epoch_time))
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
    with args.rst_client.write(args.rst_path, encoding='utf-8', overwrite=True) as writer:
        df.to_csv(writer, index=False)

    logger.info(f'Saving model at {args.model_path}')
    with args.model_client.write(args.model_path, overwrite=True) as writer:
        pkl.dump(model.state_dict(), writer)



def run(args):
    logger.info(f'Loading dataset {args.dataset} from {args.root_dir}')
    train_loader, features, n_features, num_nodes, num_edges = get_data(args, logger, mode='train')

    model = BatchModel(n_features, args.n_hidden, args.embed_dim, args.n_layers).to(args.device)
    opt = torch.optim.Adam(model.parameters())
    
    logger.info('Begin training with %d nodes, %d edges.', num_nodes, num_edges)
    train_model(args, model, train_loader, features, opt)


def config2args(config, args):
    # args.dataset = config['spaceId']
    args.outfile_path = config['outFilePath']
    args.model_path = config['modelPath']
    args.feature_names = config['featureNames']
    txt = config['flinkFeatureNames']
    args.named_feats = 'all' #[ord(s.lower())-ord('a') for s in txt if ord('A') <= ord(s) <=ord('z')] if txt!='all' else 'all'
    # args.timespan_start = int(config['startTime'])
    # args.timespan_end = int(config['endTime'])
    args.timespan_start = 20733
    args.timespan_end = 364094
    args.dgl_sampler = config['dgl_sampler']
    # args.root_dir = config['dataPath']
    return args


def train(config):
    args = easydict.EasyDict({
        'dataset': 'ia_contact', 
        'root_dir': './', 
        'prefix': 'TemporalSAGE', 
        'epochs': 2, 
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
        'cpp_file': "./wart-servers/examples/sampler.wasm", 
        'dgl_sampler': False
    })

    logger = set_logger()
    args.device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() else torch.device('cpu')
    print(args.device)
    args = config2args(config, args)
    logger.info(args)

    # PARAM_STR = f'{args.epochs}-{args.bs}-{args.num_ts}-{args.n_hidden}'
    # PARAM_STR += f'-{args.embed_dim}-{args.n_layers}-{args.lr}'
    # PARAM_STR += f'-{args.temporal_feat}-{args.named_feats}'
    # SAVE_PATH = f'{args.prefix}-{PARAM_STR}-{args.timespan_start}-{args.timespan_end}-{args.dataset}'
    
    client_path, file_path = split_url(args.outfile_path)
    args.rst_client = Client(client_path)
    args.rst_path = os.path.join(file_path, 'train_metrics.csv')

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
    parser.add_argument("--config", "-c", type=str, default='http://192.168.1.13:9009/dev/conf/train.json')
    parser.add_argument("--dgl_sampler", "-s", action='store_true')
    args = parser.parse_args()

    config = get_config(args.config)
    print(config)
    # config = {
    #     "taskId": "585838793082061314TSN",
    #     # "spaceId": "dblp-coauthors",
    #     "spaceId": "DBLPV13",
    #     # "outFilePath": "./results/",
    #     # "modelPath": "./saved_models/dblp-coauthors_2epochs.pth",
    #     "outFilePath": "http://192.168.1.13:9009/dev/pytorch/train-4FB003D56AAC/",
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
    outfile_path = train(config)
    print('outfile_path: ', outfile_path)