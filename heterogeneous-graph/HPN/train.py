import numpy as np
import torch
from prototypical_loss import prototypical_loss as loss_fn
from protonet import Heteproto
from parser_util import get_parser
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from itertools import combinations
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import time

from aminer_utils import *
from model import HGCN

def traverseList(nestList):
    '''
    将多维list转换为一维list
    '''
    flatList = []
    for item in nestList:
        if isinstance(item, list):
            flatList.extend(traverseList(item))
        else:
            flatList.append(item)
    return flatList

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def init_protonet(opt):
    '''
    Initialize the ProtoNet
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = Heteproto().to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def train(epoch, id):

	model.train()
	optimizer.zero_grad()
	logits,_,_ = model(ft_dict, adj_dict)
	#print(embd['p'])

	a_logits = F.log_softmax(logits['p'], dim=1)
	x_train_a = a_logits[id]
	y_train_a = labels_local[id]
	#y_train_a = labels_local[id].cuda()
	print(x_train_a)
	loss_train = F.nll_loss(x_train_a, y_train_a)
	f1_micro_train = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='micro')
	f1_macro_train = f1_score(y_train_a.data.cpu(), x_train_a.data.cpu().argmax(1), average='macro')

	loss_train.backward()
	optimizer.step()
	
	if epoch % 1 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1 a: {:.4f}'.format(f1_micro_train.item()),
			  'train macro f1 a: {:.4f}'.format(f1_macro_train.item()),
			 )


def test():
	model.eval()
	logits, embd, meta = model(ft_dict, adj_dict)

	return embd,meta


def meta_train(opt, features, labels, train_label, val_label, model, optim, lr_scheduler, adj, meta, nei):
	'''
    Train the model with the prototypical learning algorithm
    '''
	if val_label is None:
		best_state = None

	train_loss = []
	train_acc = []
	val_loss = []
	val_acc = []
	best_acc = 0

	best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
	last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

	for epoch in range(opt.epochs):
		print('=== Epoch: {} ==='.format(epoch))
		labels_local = labels.clone().detach()
		#task sampling
		num = [0,0,0,0,0,0,0,0]
		att = meta['p'][1:]
		for i in range(len(train_label)):
			num[i] = nei['a'][i]*att[0]+nei['c'][i]*att[1]
		num = list(softmax(num))
		select_class = np.random.choice(train_label, 2, replace=False, p=num)
		# select_class = random.sample(train_label, 2)
		print('ITERATION {} Train_Label: {}'.format(epoch + 1, select_class))

		class1_idx = []
		class2_idx = []

		for k in range(127623):
			if (labels_local[k] == select_class[0]):
				class1_idx.append(k)
			elif (labels_local[k] == select_class[1]):
				class2_idx.append(k)
		model.train()
		# for batch in tqdm(tr_iter):
		for m in range(50):
			optim.zero_grad()
			class1_train = random.sample(class1_idx, 100)
			class2_train = random.sample(class2_idx, 100)
			train_idx = class1_train + class2_train
			random.shuffle(train_idx)

			model_output = model(features)
			x_output = model_output['p'][train_idx]
			a_output = model_output['a']
			c_output = model_output['c']
			y_train = labels_local[train_idx]
			loss, acc = loss_fn(x_output,a_output,c_output,meta, adj, target=y_train,
								n_support=opt.num_support_tr)
			loss.backward()
			optim.step()
			train_loss.append(loss.item())
			train_acc.append(acc.item())
		avg_loss = np.mean(train_loss[-opt.iterations:])
		avg_acc = np.mean(train_acc[-opt.iterations:])
		print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
		lr_scheduler.step()
		if val_label is None:
			continue

		class1_val = []
		class2_val = []
		for k in range(127623):
			if (labels_local[k] == val_label[0]):
				class1_val.append(k)
				labels_local[k] = val_label[0]

			elif (labels_local[k] == val_label[1]):
				class2_val.append(k)
				labels_local[k] = val_label[1]

		model.eval()
		for m in range(50):
			val1_train = random.sample(class1_val, 100)
			val2_train = random.sample(class2_val, 100)
			val_idx = val1_train + val2_train
			random.shuffle(val_idx)
			model_output = model(features)
			xv_output = model_output['p'][val_idx]
			av_output = model_output['a']
			cv_output = model_output['c']
			yv_train = labels_local[val_idx]
			loss, acc = loss_fn(xv_output, av_output,cv_output,meta, adj, target=yv_train,
								n_support=opt.num_support_val)
			val_loss.append(loss.item())
			val_acc.append(acc.item())
		avg_loss = np.mean(val_loss[-opt.iterations:])
		avg_acc = np.mean(val_acc[-opt.iterations:])
		postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
			best_acc)
		print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
			avg_loss, avg_acc, postfix))
		if avg_acc >= best_acc:
			torch.save(model.state_dict(), best_model_path)
			best_acc = avg_acc
			best_state = model.state_dict()

	torch.save(model.state_dict(), last_model_path)

	return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def meta_test(opt, features, labels,test_idx, model, adj,meta):
	'''
    Test the model trained with the prototypical learning algorithm
    '''
	avg_acc = list()
	for epoch in range(10):
		labels_local = labels.clone().detach()

		train_idx = test_idx
		random.shuffle(train_idx)

		model_output = model(features)
		x_output = model_output[train_idx]
		a_output = model_output['a']
		c_output = model_output['c']
		y_train = labels_local[train_idx]
		loss, acc = loss_fn(x_output, a_output,c_output,meta, adj,target=y_train,
							n_support=opt.num_support_val)
		avg_acc.append(acc.item())
	avg_acc = np.mean(avg_acc)
	print('Test Acc: {}'.format(avg_acc))

	return avg_acc

if __name__ == '__main__':

	options = get_parser().parse_args()
	if not os.path.exists(options.experiment_root):
		os.makedirs(options.experiment_root)

	#cuda = False # Enables CUDA training.
	lr = 0.01 # Initial learning rate.c
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean

	adj_dict, ft_dict, labels, num_classes = load_Aminer()

	class_label = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	combination = list(combinations(class_label, 2))

	run = 1
	for i in range(len(combination)):
		print('Cross_Validation: ', i + 1)
		test_label = list(combination[i])
		train_label = [n for n in class_label if n not in test_label]
		print('Cross_Validation {} Train_Label_List {}: '.format(i + 1, train_label))
		print('Cross_Validation {} Test_Label_List {}: '.format(i + 1, test_label))

		labels_local = labels.clone().detach()
		id = []
		for k in range(127623):
			for i in range(len(train_label)):
				if (labels_local[k] == train_label[i]):
					id.append(k)
					labels_local[k] = i

		adj_pa = adj_dict['p']['a']
		adj_pc = adj_dict['p']['c']
		adj_ap = adj_dict['a']['p']
		adj_cp = adj_dict['c']['p']
		class1 = []
		class2 = []
		nei_a = []
		nei_c = []
		nei = dict()
		nei['a'] = [0,0,0,0,0,0,0,0]
		nei['c'] = [0,0,0,0,0,0,0,0]
		for k in range(127623):
			if (labels[k] == test_label[0]):
				class1.append(k)
			elif (labels[k] == test_label[1]):
				class2.append(k)
		class1 = random.sample(class1, 100)
		class2 = random.sample(class2, 100)
		test_idx = class1 + class2
		for idx in test_idx:
			nei_a.append(torch.squeeze(adj_pa[idx]._indices()).tolist())
			nei_c.append(torch.squeeze(adj_pc[idx]._indices()).tolist())
		nei_a = list(set(traverseList(nei_a)))
		nei_c = list(set(traverseList(nei_c)))
		na = []
		for idx in nei_a:
			na.append(torch.squeeze(adj_ap[idx]._indices()).tolist())
		na = list(set(traverseList(na)))
		for aa in na:
			for i in range(len(train_label)):
				if (labels_local[aa] == train_label[i]):
					nei['a'][i]+=1
		nc = []
		for idx in nei_c:
			nc.append(torch.squeeze(adj_cp[idx]._indices()).tolist())
		nc = list(set(traverseList(nc)))
		for cc in nc:
			for i in range(len(train_label)):
				if (labels_local[cc] == train_label[i]):
					nei['c'][i]=+1

		t_start = time.time()
		seed = run

		np.random.seed(seed)
		torch.manual_seed(seed)
		if options.cuda and torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)

		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)


		label = dict()
		label['p'] = labels_local
		hid_layer_dim = [64,64,64,64]
		epochs = 2
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 8)


		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)


		# Model and optimizer
		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
		model = HGCN(
					node_type = list(ft_dict.keys()),
					net_schema=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()),
					type_fusion=type_fusion,
					type_att_size=type_att_size,
					)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


		if options.cuda and torch.cuda.is_available():
			model.cuda()

			for k in ft_dict:
				ft_dict[k] = ft_dict[k].cuda()
			for k in adj_dict:
				for kk in adj_dict[k]:
					adj_dict[k][kk] = adj_dict[k][kk].cuda()
			for i in range(len(labels_local)):
				labels_local[i] = labels_local[i].cuda()

		for epoch in range(epochs):
			train(epoch, id)

		embd,meta = test()

		meta_model = init_protonet(options)
		optim = init_optim(options, meta_model)
		lr_scheduler = init_lr_scheduler(options, optim)
		res = meta_train(opt=options,
					features=embd,
					labels=labels_local,
					train_label=train_label,
					val_label=test_label,
					model=meta_model,
					optim=optim,
					lr_scheduler=lr_scheduler,
					adj=adj_dict,
				    meta=meta,
					nei=nei)
		best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
		with open('Aminer.txt', 'a') as f:
			f.write('Cross_Validation: {} Meta-Test_Accuracy: {}'.format(i + 1, best_acc))
			f.write('\n')
		print('Testing with last model..')
		meta_test(opt=options,
			 features=embd,
			 labels=labels_local,
			 test_idx=test_idx,
			 model=meta_model,
			 adj=adj_dict,
			 meta=meta)

		meta_model.load_state_dict(best_state)
		print('Testing with best model..')
		meta_test(opt=options,
			 features=embd,
			 labels=labels_local,
			 test_idx=test_idx,
			 model=meta_model,
			 adj=adj_dict,
			 meta=meta)

		t_end = time.time()
		print('Total time: ', t_end - t_start)