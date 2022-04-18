import numpy as np
import os
import pandas as pd
from random import shuffle


def _load_data(dataset="ia-contact", mode="format_data", root_dir="./"):
    edges = pd.read_csv("{}/{}/{}.edges".format(root_dir, mode, dataset))
    nodes = pd.read_csv("{}/{}/{}.nodes".format(root_dir, mode, dataset))
    return edges, nodes
    

def load_data(dataset="ia-contact", mode="format", root_dir="./"):
    """We split dataset into two files: dataset.edges, and dataset.nodes.

    """
    # Load edges and nodes dataframes from the following directories.
    # Return: a tuple of (edges, nodes) of required datasets.
    # format_data/train_data/valid_data/test_data
    # label_train_data/label_valid_data/label_test_data
    mode = "{}_data".format(mode)
    return _load_data(dataset=dataset, mode=mode, root_dir=root_dir)


def _iterate_datasets(dataset="all", mode="format_data", root_dir="./"):
    if dataset != "all":
        if isinstance(dataset, str):
            return [dataset]
        elif isinstance(dataset, list) and isinstance(dataset[0], str):
            return dataset
    fname = [f for f in os.listdir(os.path.join(root_dir, mode)) if f.endswith(".edges")]
    fpath = [os.path.join(root_dir, mode, f) for f in fname]
    lines = [len(open(f, "r").readlines()) for f in fpath]
    # sort the dataset by data size
    forder = [f[:-6] for l, f in sorted(zip(lines, fname))]
    if dataset != "all":
        if isinstance(dataset, int):
            return forder[dataset]
        elif isinstance(dataset, list) and isinstance(dataset[0], int):
            return [forder[i] for i in dataset]
        else:
            raise NotImplementedError
    return forder


def load_split_edges(dataset="ia-contact", root_dir="./"):
    train_edges, nodes = load_data(dataset=dataset, mode="train", root_dir=root_dir)
    valid_edges, _ = load_data(dataset=dataset, mode="valid", root_dir=root_dir)
    test_edges, _ = load_data(dataset=dataset, mode="test", root_dir=root_dir)
    return train_edges, valid_edges, test_edges, nodes


def load_label_edges(dataset="ia-contact", root_dir="./"):
    train_edges, nodes = load_data(dataset=dataset, mode="label_train", root_dir=root_dir)
    valid_edges, _ = load_data(dataset=dataset, mode="label_valid", root_dir=root_dir)
    test_edges, _ = load_data(dataset=dataset, mode="label_test", root_dir=root_dir)
    return train_edges, valid_edges, test_edges, nodes
    

def load_graph(dataset=None):
    """Concat the temporal edges, transform into nstep time slots, and return 
       edges, pivot_time.
    """
    train_edges, val_edges, test_edges, nodes = \
        load_split_edges(dataset=dataset)
    val_time = val_edges["timestamp"].min()
    test_time = test_edges["timestamp"].min()

    edges = pd.concat([train_edges, val_edges, test_edges])
    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    return edges, len(nodes), val_time, test_time

def load_pad_graph(dataset=None, null_idx=0):
    train_edges, val_edges, test_edges, nodes = \
        load_split_edges(dataset=dataset)
    val_time = val_edges["timestamp"].min()
    test_time = test_edges["timestamp"].min()

    edges = pd.concat([train_edges, val_edges, test_edges])
    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    edges["from_node_id"] = edges["from_node_id"].map(id2idx)
    edges["to_node_id"] = edges["to_node_id"].map(id2idx)
    
    pad = pd.DataFrame(columns=edges.columns)
    pad.loc[0] = [0] * len(edges.columns)
    pad = pad.astype(edges.dtypes)

    pad_edges = pd.concat([pad, edges], axis=0).reset_index(drop=True)
    return pad_edges, len(nodes) + 1, val_time, test_time

def load_label_data(dataset=None):
    train_edges, val_edges, test_edges, nodes = \
        load_label_edges(dataset=dataset)
    pivot_time = train_edges["timestamp"].max()

    # padding node is 0, so add 1 here.
    id2idx = {row.node_id: row.id_map + 1 for row in nodes.itertuples()}
    ans = []
    for df in [train_edges, val_edges, test_edges]:
        df["from_node_id"] = df["from_node_id"].map(id2idx)
        df["to_node_id"] = df["to_node_id"].map(id2idx)
        df = df[["from_node_id", "to_node_id", "timestamp", "label"]]
        df.columns = ["u", "i", "ts", "label"]
        ans.append(df)
    return ans[0], ans[1], ans[2]
