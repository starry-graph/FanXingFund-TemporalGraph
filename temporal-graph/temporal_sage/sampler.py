import grpc
from rpc_client import *
import time

import numpy as np
import pandas as pd
import torch
import dgl
if dgl.__version__ > '0.8.0':
    import dgl as transform
else:
    from dgl.dataloading import transform

from typing import *
from nebula_util import QueryGraphChannel


def to_pd(table) -> pd.DataFrame:
    data = {}
    for h, s in zip(table.headers, table.columns):
        if s.bool_values.data:
            series = list(s.bool_values.data)
            data[h] = pd.Series(series, dtype=np.bool8)
        elif s.int32_values.data:
            series = list(s.int32_values.data)
            data[h] = pd.Series(series, dtype=np.int32)
        elif s.int64_values.data:
            series = list(s.int64_values.data)
            data[h] = pd.Series(series, dtype=np.int64)
        elif s.float32_values.data:
            series = list(s.float32_values.data)
            data[h] = pd.Series(series, dtype=np.float32)
        elif s.float64_values.data:
            series = list(s.float64_values.data)
            data[h] = pd.Series(series, dtype=np.float64)
        elif s.string_values.data:
            series = list(s.string_values.data)
            data[h] = pd.Series(series, dtype=np.str0)
        else:
            return table.comment, None
    return table.comment, pd.DataFrame(data)


def streaming_run_iter(token: str, args_lst: List[List[str]]):
    # 第一个请求必须是Config
    yield StreamingRunRequest(
        config = StreamingRunRequest.Config(
            token = token,
        )
    )
    
    # 剩下的请求必须是Args
    for args in args_lst:
        yield StreamingRunRequest(
            args = StreamingRunRequest.Args(
                args = args
            )
        )


class MyMultiLayerSampler:   
    def __init__(self, fanouts, num_nodes, client_address = "192.168.1.11:6066", \
        cpp_file = "./sampler.wasm", graph_name='DBLPV13'):
        with open(cpp_file, "rb") as f: # 获取编译好的wasm字节码
            program = f.read()

        channel = grpc.insecure_channel(client_address) # 连接采样服务器
        self.stub = WartWorkerStub(channel) # 创建采样客户端

        # 启动采样session，设置图空间名称，上传采样脚本
        resp = self.stub.OpenSession(OpenSessionRequest(
            space_name = f"nebula:{graph_name}",
            program = program,
            # io_timeout = 1000, # 单次IO限时(废弃)
            ex_timeout = 5000, # 单次采样限时
            parallel = 64 # 并行执行
        ))
        self.token = resp.ok.token # 获得session的token

        self.fanouts = fanouts
        self.num_layer = len(fanouts)
        self.num_nodes = num_nodes

        self.clear_resp_metrics()
    
    def clear_resp_metrics(self):
        self.resp_start_times = []
        self.resp_end_times = []
        self.resp_query_counts = []
        self.resp_node_counts = []

    def fetch_neighbors(self, resp):
        # 将采样服务器的返回值转换为邻居节点
        for t in resp.tables: # test.cpp 只传回一个table，因此返回第一个table即可
            name, table = to_pd(t)
            return table    
    

    def sample_neighbors(self, year_range, seed_nodes, fanout):
        src_l, tgt_l, ts_l = [], [], []
        seed_nodes = seed_nodes.tolist() if isinstance(seed_nodes, torch.Tensor) else seed_nodes
        args = [ [str(item)] for item in seed_nodes]
        cnt = 0
        for i, resp in enumerate(self.stub.StreamingRun(streaming_run_iter(self.token, args))):
            self.resp_start_times.append(resp.sta_time)
            self.resp_end_times.append(resp.end_time)
            self.resp_query_counts.append(1)
            self.resp_node_counts.append(1)
            # self.resp_counters.append(resp.counter)
            # print(resp.logs)

            tb = self.fetch_neighbors(resp)
            try:
                df = tb[(tb.time_stamp >= year_range[0]) & (tb.time_stamp < year_range[1])]
            except: # tb is None
                continue
            src = seed_nodes[cnt]
            cnt += 1
            
            srcs = np.array([src] * len(df))
            tgts = np.array(df['target'])
            ts = np.array(df['time_stamp'])

            if fanout < len(srcs):
                idx = np.random.randint(0, len(srcs), fanout)
                srcs, tgts, ts = srcs[idx], tgts[idx], ts[idx]

            src_l.append(srcs)
            tgt_l.append(tgts)
            ts_l.append(ts)
            
            if i == len(args) - 1:
                break
        
        src_edge = [] if src_l == [] else torch.tensor(np.hstack(src_l)).to(torch.int64)
        tgt_edge = [] if src_l == [] else torch.tensor(np.hstack(tgt_l)).to(torch.int64)
        ts_edata = [] if src_l == [] else np.hstack(ts_l)

        ret = dgl.graph((tgt_edge, src_edge), num_nodes = self.num_nodes)
        ret.edata['ts'] = torch.tensor(ts_edata).to(torch.int64)
        return ret


    def sample_frontier(self, block_id, year_range, seed_nodes):
        fanout = self.fanouts[block_id]
        frontier = self.sample_neighbors(year_range, seed_nodes, fanout)
        return frontier


    def sample_blocks(self, year_range, seed_nodes):
        blocks = []
        for block_id in reversed(range(self.num_layer)):
            # print(block_id, self.fanouts[block_id], seed_nodes)
            frontier = self.sample_frontier(block_id, year_range, seed_nodes)
            block = transform.to_block(frontier, seed_nodes)
            # seed_nodes = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes} # ntype: _N
            seed_nodes = block.srcnodes['_N'].data[dgl.NID]

            blocks.insert(0, block)
        return blocks


class NeublaMultiLayerSampler:
    def __init__(self, fanouts, num_nodes, client_address="192.168.1.11:9669", graph_name='DBLPV13'):
        self.channel = QueryGraphChannel([client_address])
        self.params = [
            {
                "edge_type": "coauthor",
                # "limit": 20,
                "direct": "dst_to_src",
                # "direct": "src_to_dst",
            }
        ] # * 2
        self.space_name = graph_name
        self.node_attrs = ["author_id", "label"]

        self.fanouts = fanouts
        self.num_layer = len(fanouts)
        self.num_nodes = num_nodes

        self.clear_resp_metrics()

    def clear_resp_metrics(self):
        self.resp_start_times = []
        self.resp_end_times = []
        self.resp_query_counts = []
        self.resp_node_counts = []

    def sample_neighbors(self, year_range, seed_nodes, fanout):
        seed_nodes = torch.LongTensor(seed_nodes)
        batch_size = 100
        src_l, tgt_l, ts_l = [], [], []
        min_year, max_year = min(year_range), max(year_range)
        # print('Begin sampling with {} nodes.'.format(len(seed_nodes)))
        # print(seed_nodes)
        for batch_start in range(0, len(seed_nodes), 100):
            self.resp_start_times.append(time.time() * 1e3)

            batch_end = batch_start + batch_size
            batch_nodes = seed_nodes[batch_start:batch_end]
            resp = self.channel.sample_subgraph(self.space_name, batch_nodes, self.node_attrs, self.params)
            # 4 columns: src, dst, dist, timestamp
            edge_index = resp["edge_index"]
            src, tgt, ts = edge_index[0], edge_index[1], edge_index[3]
            ts_mask = torch.logical_and(ts >= min_year, ts <= max_year)
            src_l.append(src[ts_mask])
            tgt_l.append(tgt[ts_mask])
            ts_l.append(ts[ts_mask])


            self.resp_end_times.append(time.time() * 1e3)
            self.resp_query_counts.append(len(batch_nodes))
            self.resp_node_counts.append(len(batch_nodes))

        src_edge = [] if src_l == [] else torch.cat(src_l).to(torch.int64)
        tgt_edge = [] if src_l == [] else torch.cat(tgt_l).to(torch.int64)
        ts_edata = [] if src_l == [] else torch.cat(ts_l).to(torch.int64)

        # node_set = set(seed_nodes.tolist())
        # # src_mask = [idx.item() in node_set for idx in src_edge]
        # src_mask = torch.ones_like(src_edge)
        # src_edge = src_edge[src_mask]
        # tgt_edge = tgt_edge[src_mask]
        # ts_edata = ts_edata[src_mask]
        # print("Get {}/{} edges.".format(len(src_edge), len(src_mask)))

        # for idx in src_edge:
        # for idx in tgt_edge:
            # assert idx.item() in node_set

        ret = dgl.graph((tgt_edge, src_edge), num_nodes = self.num_nodes)
        ret.edata['ts'] = ts_edata.clone().detach()
        return ret
    
    def sample_frontier(self, block_id, year_range, seed_nodes):
        fanout = self.fanouts[block_id]
        frontier = self.sample_neighbors(year_range, seed_nodes, fanout)
        return frontier


    def sample_blocks(self, year_range, seed_nodes):
        blocks = []
        for block_id in reversed(range(self.num_layer)):
            # print(block_id, self.fanouts[block_id], seed_nodes)
            frontier = self.sample_frontier(block_id, year_range, seed_nodes)
            block = transform.to_block(frontier, seed_nodes)
            # seed_nodes = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes} # ntype: _N
            seed_nodes = block.srcnodes['_N'].data[dgl.NID]

            blocks.insert(0, block)
        return blocks

if __name__ == "__main__":
    # sampler = MyMultiLayerSampler([15, 10], num_nodes=36288, cpp_file = "../wart-servers/examples/sampler.wasm")
    
    # ret = sampler.sample_neighbors((0, 9999), [24004], fanout=15)
    # print(ret)

    # block = sampler.sample_blocks((0, 9999), [1007])
    # print(block)
    # dataloader = torch.DataLoader(train_data)
    # nfeat = xxx
    # model = xxx
    # for i, (source_nodes, target_nodes) in enumerate(dataloader):
    #     source_blocks = sampler.sample(source_nodes)
    #     target_blocks = sampler.sample(target_nodes)
    #     pred_y = model(source_blocks, target_blocks, nfeat)
    #     loss = loss_fn(pred_y, y)
    seed_nodes = [ 41,  10,   8,  30,  22,  31,  40,   2,  29,  37,  32,   0,  39,  47,
          4,  43,  36,   1,  19,  12,  44,   5,  26, 207, 199, 168, 215,   9,
        177, 127, 130,  27, 242, 220,  11, 186, 270,  64,  51,  86, 114,  62,
         87, 263, 200,  90, 176,  99, 226, 209,  66,  80,   7, 203, 219, 100,
        253,  42, 180, 108,  73, 257, 165,  13, 201, 191,  14, 119, 181, 121,
        273, 249,  45, 174,  46,  33, 223, 269,  94, 158, 259, 122, 159, 197,
        272, 229, 208, 143, 134, 111,  55, 184,  60,  16, 115,  57, 183, 146,
         65,  81, 107,  72, 211, 110, 128,  93, 145, 218, 194, 150, 246, 178,
        132, 195, 135,  52, 267, 141,  50, 227, 129, 163,  63, 109]
    sampler = NeublaMultiLayerSampler([15], num_nodes=274, graph_name='ia_contact')
    # sampler = NeublaMultiLayerSampler([15], num_nodes=36288, graph_name="DBLPV13")
    # ret = sampler.sample_neighbors((0, 9999), [24004], fanout=15)
    start = time.time()
    ret = sampler.sample_neighbors((0, 1e11), torch.arange(274), fanout=15)
    print(ret)
    print('Cost {:.2f} seconds.'.format(time.time() - start))
    # block = sampler.sample_blocks((0, 9999), [1007])
    start = time.time()
    block = sampler.sample_blocks((0, 1e11), torch.arange(274))
    print(block)
    print('Cost {:.2f} seconds.'.format(time.time() - start))
