## Temporal-Graph

### Requirements

```
numpy==1.19.5
pandas==1.1.5
scikit-learn==0.22.1
scipy==1.5.3
six==1.15.0
torch=1.5.1
tqdm==4.36.1
gpustat==0.6.0
numba==0.53.1
dgl-cu102==0.4.3post2
```

### Dataset
In this experiment, we refer to temporal graph as a graph with temporal interactions between nodes. We do not take the increase and decrease of nodes into account for technical reasons. Temporal graph datasets are downloaded from [network repository](http://networKrepository.com/ia.php) and [Snap](http://snap.stanford.edu/jodie/). Details of the datasets are listed below.

Additionally, we split the datasets into three-fold: train, valid and test datasets with 70:15:15 along the time dimension. We remove unseen nodes in the train dataset from the valid and test datasets. We generate a negative sample for each reserved edge by replacing the `to_node_id` with another random node.

| Dataset                     | Bipartite | Edge Type   | Density | Nodes | Edges  | Action Repetition | Test set | Train/Unseen nodes | d_max | d_avg  | Timespan(days) |
| --------------------------- | --------- | ----------- | ------- | ----- | ------ | ----------------- | -------- | ------------------ | ----- | ------ | -------------- |
| ia-workplace-contacts       | False     | Interaction | 2.34    | 92    | 9.8K   | 77.1%             | 4.8K     | 91/1               | 1.1K  | 106.8  | 11.43          |
| ia-contacts_hypertext2009   | False     | Proximity   | 3.28    | 113   | 20.8K  | 59.0%             | 9.9K     | 111/2              | 1.5K  | 184.2  | 2.46           |
| ia-contact                  | False     | -           | 0.75    | 274   | 28.2K  | 6.9%              | 11.2K    | 188/86             | 2.1K  | 103.1  | 3.97           |
| fb-forum                    | False     | -           | 0.08    | 899   | 33.7K  | 20.8%             | 15.7K    | 834/65             | 1.8K  | 37.51  | 164.49         |
| soc-sign-bitcoinotc         | False     | -           | 0.002   | 5.8K  | 35.5K  | 0.0%              | 6.4K     | 4.4K/1.4K          | 1.2K  | 6.05   | 1903.27        |
| ia-enron-employees          | False     | -           | 4.46    | 151   | 50.5K  | 27.3%             | 33.7K    | 148/3              | 5.2K  | 334.9  | 1137.55        |
| ia-escorts-dynamic          | True      | Rating      | 0.0009  | 10K   | 50.6K  | 10.9%             | 19.9K    | 6.7K/3.3K          | 616   | 5.01   | 2232.00        |
| ia-reality-call             | False     | Call        | 0.0022  | 7K    | 52.0K  | 31.8%             | 1.0K     | 6.7K/86            | 3.0K  | 7.6    | 106.00         |
| ia-retweet-pol              | False     | Retweet     | 0.0003  | 19K   | 61.1K  | 4.7%              | 23.0K    | 15K/3.3K           | 1.0K  | 3.3    | 48.78          |
| ia-radoslaw-email           | False     | Email       | 5.98    | 167   | 82.9K  | 18.8%             | 41.5K    | 166/1              | 9.1K  | 496.6  | 271.19         |
| ia-movielens-user2tags-10m  | True      | Assignment  | 0.0007  | 17K   | 95.5K  | 19.9%             | 33.4K    | 12.7K/3.8K         | 6.0K  | 5.8    | 1108.97        |
| soc-wiki-elec               | False     | -           | 0.004   | 7.1K  | 107.0K | 0.2%              | 12.9K    | 5.2K/1.8K          | 1.3K  | 15.04  | 1378.34        |
| ia-primary-school-proximity | False     | Proximity   | 4.31    | 242   | 125.7K | 38.3%             | 59.2K    | 242/0              | 2.6K  | 519.7  | 1.35           |
| ia-slashdot-reply-dir       | False     | -           | 0.0001  | 51K   | 140.7K | 4.2%              | 27.5K    | 39K/12K            | 3.3K  | 2.76   | 977.36         |
| JODIE-wikipedia             | True      | -           | 0.0036  | 9.2K  | 157.4K | 79.1%             | 59.7K    | 7.4K/1.7K          | 1.9K  | 17.07  | 31.00          |
| JODIE-reddit                | True      | -           | 0.011   | 10.9K | 672.4K | 61.4%             | 323.4K   | 10.8K/140          | 58.7K | 61.22  | 31.00          |

### Preprocessing

- Module construction

```
data_loader
├── data_formatter.py
├── data_unify.py
├── data_util.py
├── __init__.py
```

- Data format

`python -m data_loader.data_formatter -d CTDNE`

`python -m data_loader.data_formatter -d JODIE`

- Training data generation

`python -m data_loader.data_unify -t [datastat|datasplit|datalabel]`

### Method

#### TAP-GNN: Temporal Aggregation and Propagation GraphNeural Networks for Dynamic Representation

- Module construction

```
torch_model
├── dataset.py
├── fast_gtc.py
├── __init__.py
├── layers.py
├── node_model.py
├── online_gtc.py
├── util_dgl.py
└── visual.py
```

`python -m torch_model.fast_gtc --display --gpu -d ia-contact`

`python -m torch_model.online_gtc --display --gpu -d ia-contact`

`python -m torch_model.node_model --display --gpu -d ia-contact`

#### DPS: Learning Dynamic Preference Structure EmbeddingFrom Temporal Networks

**sample_cache**, **gumbel_cache**

- Module construction

```
sample_model
├── fusion_edge.py
├── fusion_node.py
├── fusion.py
├── graph.py
├── gumbel_alpha.py
├── gumbel_pretrain.py
├── neighbor_loader.py
├── optimal_alpha.py
```

`python -m sample_model.optimal_alpha -d ia-contact`

`python -m sample_model.gumbel_pretrain -d ia-contact`

`python -m sample_model.fusion_edge -d ia-contact`

#### TIP-GNN: Transition Propagation Graph Neural Networks forTemporal Networks

- Module construction

```
subgraph_model
├── exper_edge_np.py
├── exper_node_np.py
├── graph.py
├── mlp.py
├── preprocess.py
├── subgnn_np.py
└── vis_attn.py
```

`python -m subgraph_model.exper_edge_np -d ia-contact`

`python -m subgraph_model.exper_node_np -d ia-contact`

#### Temporal GraphSAGE: a variant working on a sequence of graph snapshots

- Model construction

```
temporal_sage
├── batch_loader.py
├── batch_model.py
├── model.py
├── temporal_sage.py
└── util.py
```

```
pip install dgl-cu102==0.6.1
python temporal_sage/temporal_sage.py --dataset fb-forum --num_ts 128 --epochs 50
```
