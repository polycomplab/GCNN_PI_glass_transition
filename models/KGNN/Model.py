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
        num_task_specific_layers=1,
    ):
        super(KGNNModel, self).__init__()

        self.num_convs = num_convs
        self.layers_in_conv = layers_in_conv
        self.channels = channels
        self.use_jumping_knowledge = use_jumping_knowledge
        self.use_dropout = use_dropout
        self.num_targets = num_targets
        self.num_fc_layers = num_fc_layers
        assert num_task_specific_layers <= num_fc_layers  # for now
        self.num_task_specific_layers = num_task_specific_layers

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

        num_fc_input_features = self.channels * 4  # for set2set + cat
        self.pre_fc_batchnorm = torch.nn.BatchNorm1d(num_fc_input_features)

        if num_task_specific_layers > 1:  # splitting layers into shared and task-specific
            # shared part
            fc_layers = []
            out_channels = None  # in_channels // 2
            for i in range(num_fc_layers-num_task_specific_layers):
                if i == 0:
                    in_channels = num_fc_input_features
                else:
                    in_channels = out_channels
                out_channels = in_channels // 2
                fc_layers += [nn.Linear(in_channels, out_channels)]
            self.fc_layers = nn.ModuleList(fc_layers)  # shared fc layers

            self.batch_norms_for_fc = nn.ModuleList(
                [
                    torch.nn.BatchNorm1d(self.fc_layers[i].out_features)
                    for i in range(len(self.fc_layers))
                ]
            )

            # tasks(targets)-specific heads
            heads = []
            for _ in range(num_targets):
                head = {}

                head_fc_layers = []
                out_channels = None  # in_channels // 2
                for i in range(num_task_specific_layers):
                    if i == 0:
                        if num_task_specific_layers == num_fc_layers:
                            in_channels = num_fc_input_features
                        else:
                            in_channels = self.fc_layers[-1].out_features
                    else:
                        in_channels = out_channels

                    if i == num_task_specific_layers-1:
                        out_channels = 1
                    else:
                        out_channels = in_channels // 2
                    head_fc_layers += [nn.Linear(in_channels, out_channels)]
                head['fc_layers'] = nn.ModuleList(head_fc_layers)

                head['batch_norms_for_fc'] = nn.ModuleList(
                    [
                        torch.nn.BatchNorm1d(head['fc_layers'][i].out_features)
                        for i in range(num_task_specific_layers-1)
                    ]
                )
                head = nn.ModuleDict(head)
                heads.append(head)

            self.heads = nn.ModuleList(heads)
        
        else:  # same thing as before
            fc_layers = []
            out_channels = None  # in_channels // 2
            for i in range(num_fc_layers):
                if i == 0:
                    in_channels = num_fc_input_features
                else:
                    in_channels = out_channels

                if i == num_fc_layers - 1:
                    out_channels = num_targets
                else:
                    out_channels = in_channels // 2
                fc_layers += [nn.Linear(in_channels, out_channels)]
            self.fc_layers = nn.ModuleList(fc_layers)
            
            self.batch_norms_for_fc = nn.ModuleList(
                [
                    torch.nn.BatchNorm1d(self.fc_layers[i+1].in_features)
                    for i in range(num_fc_layers-1)
                ]
            )

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

        # print('x_1.shape', x_1.shape, 'x_2.shape', x_2.shape)
        x = torch.cat([x_1, x_2], dim=1)

        x = self.pre_fc_batchnorm(x)

        if self.num_task_specific_layers > 1:
            for i in range(self.num_fc_layers):

                if self.use_dropout and i == 1:
                    x = self.dropout(x)

                if i < len(self.fc_layers):
                    x = self.fc_layers[i](x)
                    x = self.batch_norms_for_fc[i](x)
                    x = F.relu(x)
                else:
                    head_fc_idx = i-len(self.fc_layers)
                    if head_fc_idx == 0:
                        heads_out = [x]*self.num_targets
                    for head_idx in range(self.num_targets):
                        head = self.heads[head_idx]
                        head_fc = head['fc_layers'][head_fc_idx]
                        heads_out[head_idx] = head_fc(heads_out[head_idx])

                        if i != self.num_fc_layers - 1:
                            head_bn = head['batch_norms_for_fc'][head_fc_idx]
                            
                            heads_out[head_idx] = head_bn(heads_out[head_idx])
                            heads_out[head_idx] = F.relu(heads_out[head_idx])
            x = torch.cat(heads_out, dim=1)


        else:  # same thing as before
            for i, fc in enumerate(self.fc_layers):
                if self.use_dropout and i == 1:
                    x = self.dropout(x)

                x = fc(x)

                if i != len(self.fc_layers) - 1:
                    x = self.batch_norms_for_fc[i](x)
                    x = F.relu(x)

        return x
