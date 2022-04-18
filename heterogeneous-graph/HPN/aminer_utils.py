import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 200,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    # args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['dataset'] = 'Aminer'
    args['exp'] = '.' + os.sep + 'output'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

# 获取labels
def fake_labels(path):  # 该文件存储的是pid：label
    labels = []
    edges_unordered_pa = np.genfromtxt("{}{}.txt".format(path, 'fake_labels'), dtype=np.int32)
    for i in range(len(edges_unordered_pa)):
        labels.append(edges_unordered_pa[i][1])
    return labels

# 构造hg图
def construct_graph(path):
    paper_ids = []
    paper_names = []
    author_ids = []
    author_names = []
    conf_ids = []
    conf_names = []
    f_3 = open(os.path.join(path, "author2id.txt"), encoding="ISO-8859-1")
    f_4 = open(os.path.join(path, "conf2id.txt"), encoding="ISO-8859-1")
    f_5 = open(os.path.join(path, "fake_labels.txt"), encoding="ISO-8859-1")
    while True:
        # 作者名字不用空格，直接全部连接，并且前面还加了一个a
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        if len(z)==1:
            continue
        identity = int(z[-1])
        author_ids.append(identity)
        name = 'a'
        for i in range(len(z)-1):
            # name = name +' '+  z[i]
            name = name + z[i]
        author_names.append(name)
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[-1])
        conf_ids.append(identity)
        name = 'c'
        for i in range(len(w)-1):
            name = name + w[i]
        conf_names.append(name)

    while True:
        v = f_5.readline()
        if not v:
            break
        v = v.strip().split()
        identity = int(v[0])
        # paper_name = 'p' + ''.join(int(v[1:]))
        paper_name = 'p' + ''.join(v[0])
        paper_ids.append(identity)
        paper_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    author_ids_invmap = {x: i for i, x in enumerate(author_ids)}  # author_id:0,author_id:1,author_id:2....
    conf_ids_invmap = {x: i for i, x in enumerate(conf_ids)}
    paper_ids_invmap = {x: i for i, x in enumerate(paper_ids)}  # paper的id也是从0开始的

    # 边的信息
    adj_dict = dict()
    adj_dict['p'] = dict()
    adj_dict['a'] = dict()
    adj_dict['c'] = dict()
    pa = sparse.lil_matrix((127623, 164473))
    pc = sparse.lil_matrix((127623, 101))
    ap = sparse.lil_matrix((164473, 127623))
    cp = sparse.lil_matrix((101, 127623))
    f_1 = open(os.path.join(path, "paper_author.txt"), "r")
    f_2 = open(os.path.join(path, "paper_conference.txt"), "r")
    '''在paper-author中1621	2562这一对映射出现问题，2562在author2id这个里面丢失了它的真实作者，所以导致下面找不到2562的映射，所以我在author2id上编了一个'''
    for x in f_1:
        x = x.split('\t')
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        pa[paper_ids_invmap[x[0]],author_ids_invmap[x[1]]] = ap[author_ids_invmap[x[1]],paper_ids_invmap[x[0]]] = 1
    for y in f_2:
        y = y.split('\t')
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        pc[paper_ids_invmap[y[0]],conf_ids_invmap[y[1]]] = cp[conf_ids_invmap[y[1]],paper_ids_invmap[y[0]]] = 1
    f_1.close()
    f_2.close()

    pa = pa.tocoo()
    pc = pc.tocoo()
    ap = ap.tocoo()
    cp = cp.tocoo()
    va = torch.FloatTensor(pa.data)
    ina = torch.LongTensor(np.vstack((pa.row, pa.col)))
    sha = pa.shape
    vb = torch.FloatTensor(pc.data)
    inb = torch.LongTensor(np.vstack((pc.row,pc.col)))
    shb = pc.shape
    vc = torch.FloatTensor(ap.data)
    inc = torch.LongTensor(np.vstack((ap.row, ap.col)))
    shc = ap.shape
    vd = torch.FloatTensor(cp.data)
    ind = torch.LongTensor(np.vstack((cp.row, cp.col)))
    shd = cp.shape

    adj_dict['p']['a'] = torch.sparse.FloatTensor(ina,va,torch.Size(sha))
    adj_dict['p']['c'] = torch.sparse.FloatTensor(inb,vb,torch.Size(shb))
    adj_dict['a']['p'] = torch.sparse.FloatTensor(inc,vc,torch.Size(shc))
    adj_dict['c']['p'] = torch.sparse.FloatTensor(ind,vd,torch.Size(shd))

    return adj_dict

def load_Aminer():
    path = './Aminer/'
    # 1. 先获取labels，是这种形式[0,1,2,4,0,1,2]
    labels = fake_labels(path)
    num_classes = 10
    ft_dict = dict()

    num_nodes = len(labels)

    labels = torch.from_numpy(np.array(labels)).long()

    # 2. 构造feature  构造一个127623*1870的特征矩阵
    rd = np.random.RandomState(888)
    fp = rd.randint(0, 2, (127623, 1870))
    fp = torch.from_numpy(fp).float()
    ft_dict['p'] = torch.Tensor(fp)
    fa = rd.randint(0, 2, (164473, 1870))
    fa = torch.from_numpy(fa).float()
    ft_dict['a'] = torch.Tensor(fa)
    fc = rd.randint(0, 2, (101, 1870))
    fc = torch.from_numpy(fc).float()
    ft_dict['c'] = torch.Tensor(fc)

    # 5. 构造图
    adj_dict = construct_graph(path)

    print('dataset loaded')

    return adj_dict, ft_dict, labels, num_classes

def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


# g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
# val_mask, test_mask = load_data(args['dataset'])
def load_data(dataset, remove_self_loop=False):
    if dataset == 'Aminer':
        return load_Aminer(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
