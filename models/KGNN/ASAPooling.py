import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_scatter import scatter_max, scatter_add

from torch_geometric.utils import softmax, to_dense_adj, dense_to_sparse, remove_self_loops
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import add_remaining_self_loops

from LEConv import LEConv


class ASAPooling(torch.nn.Module):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float, optional): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        add_self_loops (bool, optional): If set to :obj:`True`, will add self
            loops to the new graph connectivity. (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """
    def __init__(self, in_channels, ratio=0.5, GNN=None, dropout=0,
                 negative_slope=0.2, add_self_loops=False, **kwargs):
        super(ASAPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        self.gnn_score = LEConv(self.in_channels, 1)
        if self.GNN is not None:
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels,
                                         **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att.reset_parameters()
        self.gnn_score.reset_parameters()
        if self.GNN is not None:
            self.gnn_intra_cluster.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        N = x.size(0)

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x

        x_pool = x
        if self.GNN is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index,
                                            edge_weight=edge_weight)

        x_pool_j = x_pool[edge_index[0]]
        x_q = scatter_max(x_pool_j, edge_index[1], dim=0)[0]
        x_q = self.lin(x_q)[edge_index[1]]

        score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
        score = F.leaky_relu(score, self.negative_slope)
        score = softmax(score, edge_index[1], num_nodes=N)

        # Sample attention coefficients stochastically.
        score = F.dropout(score, p=self.dropout, training=self.training)

        v_j = x[edge_index[0]] * score.view(-1, 1)
        x = scatter_add(v_j, edge_index[1], dim=0)

        # Cluster selection.
        fitness = self.gnn_score(x, edge_index).sigmoid().view(-1)
        perm = topk(fitness, self.ratio, batch)
        x = x[perm] * fitness[perm].view(-1, 1)
        batch = batch[perm]

        # Graph coarsening.
        A = to_dense_adj(edge_index, edge_attr=edge_weight).squeeze(0)
        S = to_dense_adj(edge_index, edge_attr=score).squeeze(0)

        S = S[:, perm]

        A = S.t() @ A @ S  # sparse by sparse multiplication

        if self.add_self_loops:
            raise NotImplementedError()
            # A = A.fill_diag(1.)
        # else:
        #     A = A.remove_diag()
        edge_index, edge_weight = dense_to_sparse(A)
        # print('edge_index.shape', edge_index.shape)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return x, edge_index, edge_weight, batch, perm

    def __repr__(self):
        return '{}({}, ratio={})'.format(self.__class__.__name__,
                                         self.in_channels, self.ratio)