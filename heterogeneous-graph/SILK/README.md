# SILK
## Walking with Attention: Self-guided Walking for Heterogeneous Graph Embedding
Here we provide an implementation of SILK. SILK assumes no prior knowledge or annotation is provided, and conducts a customized random walk to encode the contexts of the heterogeneous graph of interest. Specifically, this is achieved via maintaining a dynamically-updated guidance matrix that records the node-conditioned transition potentials.
![SILK](figure2.pdf)
### Environment settings

- python==3.8.2
- networkx==1.11
- numpy==1.11.2
- gensim==0.13.3
### How to run example

```python
python main.py --input data/imdb --dimensions 128 --walk_length 50 --num_walks 10 --window-size 5 --alpha 0.5 --output movie_embeddings.txt
```
#### General Options

- --input the input file of a graph;
- --type the node type for classification;
- --alpha the initial revise probability;
- --epochs the number of epochs;
- --dimensions the number of diimensions;
- --walk_length the length of random walks;
- --num_walks the number of random walks;
- --window-size the window size for skip-gram model;
- --output the output file of representation;
- --clf_ratio the ratio of training data for node classification;

#### Input
The supported input format is the edgelist of a graph:

```python
edgelist: node1 node2 <weight_float, optional>
```
#### Output
The output file has n+1 lines for a graph with n nodes. The first line has the folloowing format:

```python
number_of_nodes dim_of_embedding
```
The next n lines are as follows:

```python
node_id dim1 diim2 ... dimd
```
where dim1,...,dimd is the d-dimensional embedding learned by SILK.
#### Data
In order to use your own data, you have to provide

- an |E| edge list (|E| is the number of edges),
- an |V| label list (|V| is the number of nodes).

In this example, we load IMDB dataset. The original dataset is provided by https://github.com/Jhy1993/HAN.

