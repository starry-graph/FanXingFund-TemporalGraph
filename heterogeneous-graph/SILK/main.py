import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import math
import argparse
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score
from gensim.models import Word2Vec
from walk import RWGraph
#from utils import * 
from classify import *
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='data/imdb',
                        help='Input dataset path')

    parser.add_argument('--type', type=str, default='m',
                        help='node type for classification. Default is paper.')
                        
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='node type for classification. Default is paper.')
                        
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epoch. Default is 10.')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    

    parser.add_argument('--walk_length', type=int, default=50,
                        help='Length of walk per source. Default is 100.')

    parser.add_argument('--num_walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Context size for optimization. Default is 10.')
    
    parser.add_argument('--output',type=str, default='movie_embedding.txt',
                        help='Input dataset path')
                        
    parser.add_argument('--clf_ratio', default=0.8, type=float,
                        help='The ratio of training data in the classification')
                        
    
    return parser.parse_args()

def generate_walks(G, alpha, type, type_att):
    walker = RWGraph(G)
    all, walks, paths = walker.simulate_walks(args.num_walks, args.walk_length, alpha, type, type_att)

    print('finish generating the walks')

    return all, walks, paths

def generate_pairs(walks,paths):
    pairs = []
    skip_window = args.window_size // 2
    for index in range(args.num_walks):
        for i in range(len(walks[index])):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    path = paths[index]
                    words = path[i-j:i]
                    type = max(words,key=words.count)
                    pairs.append((walks[index][i], walks[index][i - j],type))
                if i + j < len(walks[index]):
                    path = paths[index]
                    words = path[i:i+j]
                    type = max(words,key=words.count)
                    pairs.append((walks[index][i], walks[index][i + j],type))
    return pairs
    
def g_pairs(walks):
    pairs = []
    skip_window = args.window_size // 2
    for index in range(args.num_walks):
        for i in range(len(walks[index])):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walks[index][i], walks[index][i - j]))
                if i + j < len(walks[index]):
                    pairs.append((walks[index][i], walks[index][i + j]))
    return pairs

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
    
def kmeans_nmi(x, y, k):
    km = KMeans(n_clusters=k)
    km.fit(x,y)
    y_pre = km.predict(x)

    nmi = normalized_mutual_info_score(y, y_pre)
    print('NMI: {}'.format(nmi))
    
if __name__ == "__main__":
    args = parse_args()
    file_name = args.input
    print(args)
    t1 = time.time()
    G = nx.read_edgelist(file_name + '/imdb_edges.txt')
    type = args.type
    nodes = list(n for n in G.nodes() if(n[0] == type))
    X, Y = read_node_label(file_name + '/labellessid.txt')

    type_att = dict()
    for node in nodes:
        type_att[node] = [0.1,0.9]
    for epoch in range(args.epochs):
        print('epoch num:', epoch)
        all, walks, paths= generate_walks(G, args.alpha, type, type_att)
        pairs = generate_pairs(walks,paths)
        #all_pairs = g_pairs(all)

        model = Word2Vec(all, size=args.dimensions, window=args.window_size, min_count=0, negative = 5, iter = 5, workers = 5, sg=1)
        vectors = model.wv

        clf = Classifier(vectors=vectors, clf=LogisticRegression())
        clf.split_train_evaluate(X, Y, args.clf_ratio)
        
        xk= [vectors[x] for x in X]
        kmeans_nmi(xk, Y, k=5)
        type_att_new = dict()
        matrix_dis = 0
        for node in nodes:
            nei1 = []
            nei2 = []

            
            for pair in pairs:
                if pair[0] == node:
                    if pair[2] == 'd':
                        nei1.extend([vectors[pair[1]]])
                    else:
                        nei2.extend([vectors[pair[1]]])
            '''
            for pair in all_pairs:
                if pair[0] == node:
                    if pair[1][0] == 'd':
                        nei1.extend([vectors[pair[1]]])
                    else:
                        nei2.extend([vectors[pair[1]]])
            '''

            if not nei1:
                dis1 = -1
            else:
                vec1 = [np.mean(nei1, axis=0)]
                dis1 = np.dot(vec1, vectors[node]) / (np.linalg.norm(vec1) * np.linalg.norm(vectors[node]))
            if not nei2:
                dis2 = -1
            else:
                vec2 = [np.mean(nei2, axis=0)]
                dis2 = np.dot(vec2, vectors[node]) / (np.linalg.norm(vec2) * np.linalg.norm(vectors[node]))

            '''
            dis = [('d', float(dis1)),('a', float(dis2))]
            dis = sorted(dis, key=lambda dis:dis[1], reverse=True)
            type_att_new[node] = [d[0] for d in dis]
            '''
            dis = [float(dis1),float(dis2)]
            type_att_new[node] = list(softmax(dis))
        if type_att_new == type_att:
            break
        else:
            for node in nodes:
                matrix_dis += abs(float(type_att_new[node][0])-float(type_att[node][0]))
                matrix_dis += abs(float(type_att_new[node][1])-float(type_att[node][1]))
            print(matrix_dis)
            type_att = type_att_new
    t2 = time.time()
    print('time:',t2-t1)
    model.wv.save_word2vec_format(args.output)
