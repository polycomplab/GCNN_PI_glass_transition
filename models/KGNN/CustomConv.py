import math

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class CustomConv(MessagePassing):

    def __init__(self,
                 out_channels,
                 num_layers,
                 num_edge_types,
                 num_node_types,
                 aggr,
                 bias=True,
                 use_nodetype_coeffs=False,
                 use_edgetype_coeffs=False,
                 use_jumping_knowledge=False,
                 use_bias_for_update=False,
                 **kwargs):
        super(CustomConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.use_nodetype_coeffs = use_nodetype_coeffs
        self.use_edgetype_coeffs = use_edgetype_coeffs
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_bias_for_update = use_bias_for_update

        # self.weight = torch.nn.Parameter(
        #     # torch.Tensor(num_layers, num_edge_types, out_channels, out_channels)
        #     torch.Tensor(num_layers, out_channels, out_channels)
        #     # torch.Tensor(num_layers, num_node_types, out_channels, out_channels)
        # )

        self.w = nn.ModuleList(
            [nn.Linear(out_channels, out_channels) for _ in range(3)]
        )

        if self.use_edgetype_coeffs:
            self.w_edgetypes = torch.nn.Parameter(
                torch.Tensor(num_edge_types)
            )
        if self.use_nodetype_coeffs:
            self.w_nodetypes = torch.nn.Parameter(
                torch.Tensor(num_node_types)
            )
        # self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        if use_bias_for_update:
            self.bias_for_update = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_for_update', None)

        self.reset_parameters()

    def reset_parameters(self):
        # self.uniform(self.out_channels, self.weight)
        if self.use_edgetype_coeffs:
            # torch.nn.init.xavier_normal_(self.w_edgetypes)
            self.uniform(self.num_edge_types, self.w_edgetypes)
        if self.use_nodetype_coeffs:
            # torch.nn.init.xavier_normal_(self.w_nodetypes)
            self.uniform(self.num_node_types, self.w_nodetypes)
        if self.use_bias_for_update:
            torch.nn.init.zeros_(self.bias_for_update)
        # self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        ms = []
        # for i in range(self.num_layers):
            # m = torch.matmul(h, self.weight[i])
        m = self.w[0](h)
        # m = F.relu(m)

        if self.use_nodetype_coeffs:
            lambda_mask = m.new_zeros(m.shape[0])
            for j in range(self.num_node_types):
                mask = x[:, j] == 1
                lambda_mask += mask * self.w_nodetypes[j]
            m *= torch.sigmoid(lambda_mask.unsqueeze(1))

        if self.use_edgetype_coeffs:
            edge_weight = torch.matmul(edge_attr[:, :self.num_edge_types],
                                       self.w_edgetypes.unsqueeze(1))
            edge_weight = torch.sigmoid(edge_weight)
        else:
            edge_weight = None

        m_new = self.propagate(edge_index, x=m, edge_weight=edge_weight)

        # h = self.rnn(m_new, h)
        # h = h + F.relu(self.w[1](m_new))
        h = self.w[1](m + m_new)

        h = F.relu(h)

        h = self.w[2](h)

        if self.use_jumping_knowledge:
            ms.append(m_new)

        if self.use_jumping_knowledge:
            out = h, ms
        else:
            out = h
        # out = h
        return out

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def update(self, aggr_out):
        if self.use_bias_for_update:
            aggr_out = aggr_out + self.bias_for_update
        return aggr_out

