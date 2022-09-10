import torch.nn as nn
from .bayesian_gcn import BGCNConv
from torch_geometric.nn import BatchNorm


class BGCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels,is_cached=True):
        super(BGCNEncoder, self).__init__()
        self.conv1 = BGCNConv(in_channels,  out_channels, cached=is_cached) 
    
        self.norm1 = BatchNorm(out_channels)

    def forward(self, x, edge_index):
        return self.norm1(self.conv1(x, edge_index).tanh())
    def get_pw(self):
      return self.conv1.get_pw() + self.conv2.get_pw()
    def get_qw(self):
      return self.conv1.get_qw() + self.conv2.get_qw()