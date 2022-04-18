import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureHook():
    def __init__(self, module):
        self.module = module
        self.feat_in = None
        self.feat_out = None
        self.register()

    def register(self):
        self._hook = self.module.register_forward_hook(self.hook_fn_forward)

    def remove(self):
        self._hook.remove()

    def hook_fn_forward(self, module, fea_in, fea_out):
        self.feat_in = fea_in[0]
        self.feat_out = fea_out


class AmalBlock(nn.Module):
    def __init__(self, cs, ct1, ct2):
        super(AmalBlock, self).__init__()
        self.cs, self.ct1, self.ct2 = cs, ct1, ct2
        self.align1 = nn.Linear(self.cs, self.ct1, bias=True)
        self.align2 = nn.Linear(self.cs, self.ct2, bias=True)

    def forward(self, fs):
        _fst1 = self.align1(fs)
        _fst2 = self.align2(fs)
        return _fst1, _fst2
