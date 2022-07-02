import math

import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing


class DistanceConv2(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 out_channels,
                 hop_encoding_size):
        super(DistanceConv2, self).__init__()

        self.channels = channels
        self.out_channels = out_channels

        self.nn = nn.Sequential(nn.Linear(in_channels*2 + hop_encoding_size, channels),
                                nn.ReLU(),
                                nn.Linear(channels, out_channels))

    def forward(self, x, edge_index, edge_attr):
        """"""
        # x = x.unsqueeze(-1) if x.dim() == 1 else x
        # print('x shape', x.shape)
        x_1 = torch.index_select(x, 0, edge_index[0])
        x_2 = torch.index_select(x, 0, edge_index[1])
        edge_info = torch.cat([x_1, x_2, edge_attr], dim=1)
        out = self.nn(edge_info)#.view(-1, self.in_channels, self.out_channels)
        return out#self.propagate(edge_index, x=x, weight=weight)

    # def message(self, x_j, weight):
    #     # weight2 = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
    #     # print('max diff', torch.max(weight - weight2))
    #     # exit(0)
    #     # print('first shape', x_j.unsqueeze(1).shape, '2nd shape', weight.shape)
    #     # print('x_j shape', x_j.shape)
    #     return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    # def update(self, aggr_out, x):
    #     # print('aggr_out before', aggr_out.shape)
    #     # if self.root is not None:
    #     #     aggr_out = aggr_out + torch.mm(x, self.root)
    #     # if self.bias is not None:
    #     #     aggr_out = aggr_out + self.bias
    #     # print('aggr_out after ', aggr_out.shape)
    #     return aggr_out

