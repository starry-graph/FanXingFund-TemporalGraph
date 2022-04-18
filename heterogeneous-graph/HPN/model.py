import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import HeteGCNLayer



class HGCN(nn.Module):
	
	def __init__(self, node_type, net_schema, layer_shape, label_keys, type_fusion='att', type_att_size=64):
		super(HGCN, self).__init__()
		
		self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
		self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)
		self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], type_fusion, type_att_size)
		self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], type_fusion, type_att_size)


		self.embd2class = nn.ParameterDict()
		self.bias = nn.ParameterDict()
		self.label_keys = label_keys
		self.node_type = node_type
		for k in label_keys:
			self.embd2class[k] = nn.Parameter(torch.FloatTensor(layer_shape[-2][k], layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.embd2class[k].data, gain=1.414)
			self.bias[k] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1][k]))
			nn.init.xavier_uniform_(self.bias[k].data, gain=1.414)


	def forward(self, ft_dict, adj_dict):

		x_dict,_ = self.hgc1(ft_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict,_ = self.hgc2(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict,_ = self.hgc3(x_dict, adj_dict)
		x_dict = self.non_linear(x_dict)
		x_dict = self.dropout_ft(x_dict, 0.5)
		
		x_dict,meta = self.hgc4(x_dict, adj_dict)

		logits = {}
		embd = {}
		for k in self.node_type:
			embd[k] = x_dict[k]
		for k in self.label_keys:
			logits[k] = torch.mm(x_dict[k], self.embd2class[k]) + self.bias[k]
		return logits,embd,meta


	def non_linear(self, x_dict):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.elu(x_dict[k])
		return y_dict


	def dropout_ft(self, x_dict, dropout):
		y_dict = {}
		for k in x_dict:
			y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
		return y_dict
