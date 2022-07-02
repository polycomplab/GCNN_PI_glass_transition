import math

import torch
from torch_geometric.nn.conv import MessagePassing


class ModifiedGatedGraphConv(MessagePassing):

    def __init__(self,
                 out_channels,
                 num_layers,
                 num_edge_types,
                 num_node_types,
                 aggr,
                 edge_in_size,
                 bias=True,
                 use_nodetype_coeffs=False,
                 use_jumping_knowledge=False,
                 use_bias_for_update=False,
                 # use_edgeattr_data=True,
                 **kwargs):
        super(ModifiedGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.use_nodetype_coeffs = use_nodetype_coeffs
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_bias_for_update = use_bias_for_update
        # self.use_edgeattr_data = use_edgeattr_data

        self.weight = torch.nn.Parameter(
            # torch.Tensor(num_layers, num_edge_types, out_channels, out_channels)
            torch.Tensor(num_layers, out_channels, out_channels)
            # torch.Tensor(num_layers, num_node_types, out_channels, out_channels)
        )

        # if use_edgeattr_data:
        self.nn = torch.nn.Sequential(torch.nn.Linear(edge_in_size, 128),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(128, self.out_channels * self.out_channels))
        # self.nn = torch.nn.Sequential(torch.nn.Linear(4, 128),
        #                                torch.nn.ReLU(),
        #                                torch.nn.Linear(128, self.out_channels * self.out_channels))

        if self.use_nodetype_coeffs:
            self.w_nodetypes = torch.nn.Parameter(
                torch.Tensor(num_layers, num_node_types)
            )
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)

        self.batch_norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(out_channels) for _ in range(num_layers)]
        )

        if use_bias_for_update:
            self.bias_for_update = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_for_update', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.uniform(self.out_channels, self.weight)
        if self.use_nodetype_coeffs:
            torch.nn.init.xavier_normal_(self.w_nodetypes)
            # self.uniform(self.num_node_types, self.w_nodetypes)
        if self.use_bias_for_update:
            torch.nn.init.zeros_(self.bias_for_update)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        if edge_attr is not None:
            weight = self.nn(edge_attr).view(-1, self.out_channels, self.out_channels)

        ms = []
        for i in range(self.num_layers):
            m = torch.matmul(h, self.weight[i])

            if self.use_nodetype_coeffs:
                lambda_mask = m.new_zeros(m.shape[0])
                for j in range(self.num_node_types):
                    mask = x[:, j] == 1
                    lambda_mask += mask * self.w_nodetypes[i, j]
                m *= torch.sigmoid(lambda_mask.unsqueeze(1))

            # if self.use_edgeattr_data:
            if edge_attr is not None:
                m_new = self.propagate(edge_index, x=m, weight=weight)
            else:
                m_new = self.propagate(edge_index, x=m, weight=None)

            if self.use_jumping_knowledge:
                ms.append(m_new)  # last layer's output is excluded from JK output!

            m_new = self.batch_norms[i](m_new)

            h = self.rnn(m_new, h)

        if self.use_jumping_knowledge:
            out = h, ms
        else:
            out = h
        return out

    def message(self, x_j, weight): #edge_weight):#, pseudo):
        if weight is not None:
            return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        else:
            return x_j

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def update(self, aggr_out):
        if self.use_bias_for_update:
            aggr_out = aggr_out + self.bias_for_update
        return aggr_out

