## Temporal-Graph

### Requirements

- `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y`

```
numpy==1.19.5
pandas==1.1.5
scikit-learn==0.22.1
scipy==1.5.3
six==1.15.0
tqdm==4.36.1
gpustat==0.6.0
numba==0.53.1
dgl-cu111==0.6.1
easydict==1.9
hdfs==2.7.0
grpcio==1.47.0
grpcio-tools==1.47.0
pandas==1.1.5
```

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



### 配置采样算子的TemporalSAGE

#### Wart-Servers

- 环境配置：
```
# 安装rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 安装wasi-sdk >= 15.0
wget "https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-15/wasi-sdk-15.0-linux.tar.gz"
tar xvf wasi-sdk-15.0-linux.tar.gz -C ${HOME}

# 安装python环境
pip install grpcio grpcio-tools pandas
```

- 启动采样服务器：
```
cd temporal-graph/
git clone https://github.com/wjie98/wart-servers.git
cd wart-servers/wart-worker

# 编辑配置文件，修改端口
vim config.yaml

# 启动, 等待服务器运行
cargo run --release --bin server config.yaml
```

- 编译
```
# 打开另一个终端
cd temporal-graph/wart-servers/examples/
../wasm/build.sh

# 编辑编译文件 build.sh，修改 OUT_PATH="../../temporal_sage/rpc_client"
# 生成python客户端
cp rpc_client/ ../../temporal_sage/ -r
./build.sh

# 复制并编译采样脚本，生成二进制文件 sampler.wasm
cp ../../temporal_sage/sampler.cpp ./
./wacc sampler.cpp
cd ../../
```

#### 运行TemporalSAGE

- Ensure dgl < 0.8.0
```
pip install dgl-cu111==0.6.1
# or pip install dgl-cu102==0.6.1
```

- Use wart-Servers sampler:
```
python temporal_sage/train.py -c http://192.168.1.13:9009/dev/conf/train.json
python temporal_sage/infer.py -c http://192.168.1.13:9009/dev/conf/infer.json
```

- Use dgl sampler:
```
python temporal_sage/train.py -c http://192.168.1.13:9009/dev/conf/train.json --dgl_sampler
python temporal_sage/infer.py -c http://192.168.1.13:9009/dev/conf/infer.json --dgl_sampler
```

### Method

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
python temporal_sage/temporal_sage.py --dataset fb-forum --num_ts 128 --epochs 50 --named_feats 1 2 3 4 5 6 7 --timespan_start 1095290000 --timespan_end 1096500000
```

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


