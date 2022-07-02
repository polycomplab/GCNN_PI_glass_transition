import math

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_scatter import scatter_add, scatter_mean
from k_gnn import avg_pool

from .ModifiedGatedGraphConv import ModifiedGatedGraphConv
from .CustomConv import CustomConv
from .DistanceConv import DistanceConv
from .DistanceConv2 import DistanceConv2
from .CustomNNConv import CustomNNConv


class KGNNModel(torch.nn.Module):
    def __init__(
        self,
        layers_in_conv=3,
        channels=64,
        use_nodetype_coeffs=True,
        num_node_types=9,
        num_edge_types=4,
        use_jumping_knowledge=False,
        embedding_size=64,
        use_bias_for_update=True,
        use_dropout=True,
        num_convs=3,
        num_fc_layers=3,
        neighbors_aggr='add',
        dropout_p=0.1,
        num_targets=1,
    ):
        super(KGNNModel, self).__init__()

        self.num_convs = num_convs
        self.layers_in_conv = layers_in_conv
        self.channels = channels
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_dropout = use_dropout

        self.mggc1 = ModifiedGatedGraphConv(
            channels,
            layers_in_conv,
            num_edge_types,
            num_node_types,
            neighbors_aggr,
            edge_in_size=10,
            # edge_in_size=5, # using both distances and bond types
            use_nodetype_coeffs=use_nodetype_coeffs,
            use_jumping_knowledge=use_jumping_knowledge,
            use_bias_for_update=use_bias_for_update,
        )

        self.mggc2 = ModifiedGatedGraphConv(
            channels,
            layers_in_conv,
            num_edge_types,
            num_node_types,
            neighbors_aggr,
            edge_in_size=10,  # for june-2021 pretraining Askadsky data and transforms not trimmed
            # edge_in_size=5, # using both distances and bond types
            use_nodetype_coeffs=use_nodetype_coeffs,
            use_jumping_knowledge=use_jumping_knowledge,
            use_bias_for_update=use_bias_for_update,
        )

        self.mggc3 = ModifiedGatedGraphConv(
            channels,
            layers_in_conv,
            num_edge_types,
            num_node_types,
            neighbors_aggr,
            edge_in_size=10,  # for june-2021 pretraining Askadsky data and transforms not trimmed
            # edge_in_size=5, # using both distances and bond types
            use_nodetype_coeffs=use_nodetype_coeffs,
            use_jumping_knowledge=use_jumping_knowledge,
            use_bias_for_update=use_bias_for_update,
        )

        # k-gnn convs
        self.mggc4 = ModifiedGatedGraphConv(
            channels,  # + 24,
            layers_in_conv,
            num_edge_types,
            num_node_types,
            neighbors_aggr,
            edge_in_size=10,  # for june-2021 pretraining Askadsky data and transforms not trimmed
            # edge_in_size=5, # using both distances and bond types
            use_nodetype_coeffs=use_nodetype_coeffs,
            use_jumping_knowledge=use_jumping_knowledge,
            use_bias_for_update=use_bias_for_update,
        )

        self.mggc5 = ModifiedGatedGraphConv(
            channels,  # + 24,
            layers_in_conv,
            num_edge_types,
            num_node_types,
            neighbors_aggr,
            edge_in_size=10,  # for june-2021 pretraining Askadsky data and transforms not trimmed
            # edge_in_size=5, # using both distances and bond types
            use_nodetype_coeffs=use_nodetype_coeffs,
            use_jumping_knowledge=use_jumping_knowledge,
            use_bias_for_update=use_bias_for_update,
        )

        self.set2set_1 = pyg.nn.Set2Set(channels, processing_steps=5, num_layers=2)
        self.set2set_2 = pyg.nn.Set2Set(channels, processing_steps=5, num_layers=2)
        # self.set2set_3 = pyg.nn.Set2Set(channels, processing_steps=5, num_layers=2)

        self.batch_norms = nn.ModuleList(
            [torch.nn.BatchNorm1d(channels) for _ in range(num_convs)]
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.fc_layers = nn.ModuleList(
            self.make_fc_layers(num_fc_layers, num_targets=num_targets)
        )

        self.pre_fc_batchnorm = torch.nn.BatchNorm1d(self.fc_layers[0].in_features)

        self.batch_norms_for_fc = nn.ModuleList(
            [
                torch.nn.BatchNorm1d(self.fc_layers[i + 1].in_features)
                for i in range(num_fc_layers - 1)
            ]
        )

    def make_fc_layers(self, num_fc_layers, num_targets):
        fc_layers = []
        in_channels = self.channels * 4  # for set2set + cat
        out_channels = None  # in_channels // 2
        for i in range(num_fc_layers):
            if i != 0:
                in_channels = out_channels
            if i == num_fc_layers - 1:
                out_channels = num_targets
            else:
                out_channels = in_channels // 2
            fc_layers += [nn.Linear(in_channels, out_channels)]
        return fc_layers

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        x = self.mggc1(x, edge_index, edge_attr)
        x = self.batch_norms[0](x)
        x = F.relu(x)

        x = self.mggc2(x, edge_index, edge_attr)
        x = self.batch_norms[1](x)
        x = F.relu(x)

        x = self.mggc3(x, edge_index, edge_attr)  # , edge_weight.unsqueeze(1))
        x = self.batch_norms[2](x)
        x = F.relu(x)

        x_1 = self.set2set_1(x, batch)

        # 2-graph part
        x = avg_pool(x, data.assignment_index_2)
        # x = torch.cat([x, data.iso_type_2], dim=1)

        x = F.relu(self.mggc4(x, data.edge_index_2))
        x = F.relu(self.mggc5(x, data.edge_index_2))

        x_2 = self.set2set_2(x, data.batch_2)

        x = torch.cat([x_1, x_2], dim=1)

        x = self.pre_fc_batchnorm(x)

        for i, fc in enumerate(self.fc_layers):
            if self.use_dropout and i == 1:
                x = self.dropout(x)

            x = fc(x)

            if i != len(self.fc_layers) - 1:
                x = self.batch_norms_for_fc[i](x)
                x = F.relu(x)

        return x
