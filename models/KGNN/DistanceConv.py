import math

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing


class DistanceConv(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 # nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(DistanceConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.nn = nn.Sequential(nn.Linear(4, 128),  # 128 is too much?
                                nn.ReLU(),
                                nn.Linear(128, in_channels * out_channels))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        # print('x shape', x.shape)
        weight = self.nn(edge_attr).view(-1, self.in_channels, self.out_channels)
        return self.propagate(edge_index, x=x, weight=weight)

    def message(self, x_j, weight):
        # weight2 = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        # print('max diff', torch.max(weight - weight2))
        # exit(0)
        # print('first shape', x_j.unsqueeze(1).shape, '2nd shape', weight.shape)
        # print('x_j shape', x_j.shape)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        # print('aggr_out before', aggr_out.shape)
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        # print('aggr_out after ', aggr_out.shape)
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

