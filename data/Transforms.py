from collections import defaultdict
import time

import torch
import torch_geometric as pyg
import torch_geometric.transforms as pyg_transforms
from k_gnn import TwoMalkin


# Connects all nodes in a graph
class ConnectAll(object):
    def __init__(self, loop=False, max_num_neighbors=32, flow="source_to_target"):
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow

    def __call__(self, data):
        tensors = []
        seq = torch.arange(0, data.num_nodes, dtype=torch.int64)
        for i in range(seq.size(0)):
            a = (seq != i).nonzero().t()
            b = torch.full_like(a, fill_value=i)
            t = torch.cat((a, b), dim=0)
            tensors.append(t)
        new_edge_index = torch.cat(tensors, dim=1)
        new_edge_attr = torch.zeros(
            (new_edge_index.size(1), data.num_edge_features), dtype=torch.float32
        )

        for index, edge in enumerate(data.edge_index.split(1, dim=1)):
            new_index = torch.nonzero((new_edge_index == edge).all(dim=0)).squeeze()
            new_edge_attr[new_index, :] = data.edge_attr[index, :]

        data.edge_index = new_edge_index
        data.edge_attr = new_edge_attr

        return data


class AddNHopToEdgeIndexAndAttr(object):
    def __init__(self, max_hops):
        self.max_hops = max_hops

    def my_bfs(self, i_to_j, start_node, dist_map):

        init_num_hops = 0
        visited = [start_node]
        queue = [(start_node, init_num_hops)]

        while queue:
            node, hops = queue.pop(0)
            hops += 1

            neighbors = i_to_j[node]

            for neighbor in neighbors:
                if neighbor not in visited:
                    dist_map[(start_node, neighbor)] = hops

                    visited.append(neighbor)
                    queue.append((neighbor, hops))

    def __call__(self, data):
        edges_by_hops = self.make_edges_by_hops(data)

        hops_of_edges_list = []
        for hop in range(1, self.max_hops + 1):
            edges_list = edges_by_hops[hop]
            num_edges = len(edges_list)
            if num_edges == 0:  # some molecules are too small
                continue

            new_index = data.edge_index.new_tensor(edges_list).t()
            if hop == 1:
                data.new_edge_index = new_index
            else:
                data.new_edge_index = torch.cat((data.new_edge_index, new_index), dim=1)

            new_attr_shape = (new_index.shape[1], self.max_hops)
            new_attr = data.edge_attr.new_zeros(new_attr_shape)
            new_attr[:, hop - 1] = 1

            if hop == 1:
                data.new_edge_attr = new_attr
            else:
                data.new_edge_attr = torch.cat((data.new_edge_attr, new_attr), dim=0)

            corresponding_hops = torch.full([num_edges], hop, dtype=torch.int)
            hops_of_edges_list.append(corresponding_hops)

        data.hops_of_edges = torch.cat(hops_of_edges_list, dim=0)
        return data

    def make_edges_by_hops(self, data):
        dist_map = {}
        i_to_j = (
            []
        )  # list of lists, where each sublist (at index i) is a list of nodes j connected to node i
        for node_i in range(data.num_nodes):
            mask_connected = data.edge_index[0].eq(node_i)
            nodes_j = torch.masked_select(data.edge_index[1], mask_connected).tolist()
            i_to_j.append(nodes_j)

        for start_node in range(data.num_nodes):
            self.my_bfs(i_to_j, start_node, dist_map)

        edges_by_hops = defaultdict(list)
        for k, v in sorted(dist_map.items()):
            edges_by_hops[v].append(k)
        return edges_by_hops


class AddSelfLoops(object):
    def __init__(self):
        pass

    def __call__(self, data):
        data.edge_index, _ = pyg.utils.add_self_loops(data.edge_index)
        num_self_loops = data.edge_index.size(1) - data.edge_attr.size(0)
        extra_attr = torch.zeros(
            (num_self_loops, data.num_edge_features), dtype=torch.float32
        )
        data.edge_attr = torch.cat([data.edge_attr, extra_attr], axis=0)
        return data


# In some molecules there are only hydrogen bonds,
# so add self-loops first to ensure that any molecule has bonds
class RemoveHydrogens(object):
    def __init__(self):
        pass

    def __call__(self, data):

        x_hydrogens = torch.nonzero(data.x[:, 0] == 1).squeeze()
        x2_mask = torch.nonzero(data.x[:, 0] == 0).squeeze(1)
        num_hydrogens = x_hydrogens.nelement()

        def get_non_hydrogens():
            if num_hydrogens == 0:
                non_hydrogens_mask = torch.ones(
                    data.edge_index.size(1), dtype=torch.bool
                )
            elif num_hydrogens == 1:
                non_hydrogens_mask = ~((data.edge_index == x_hydrogens).any(axis=0))
            else:
                hydrogen_masks = [
                    (data.edge_index == v).any(axis=0) for v in x_hydrogens
                ]
                non_hydrogens_mask = ~torch.stack(hydrogen_masks).any(axis=0)
            return non_hydrogens_mask

        non_hydrogens_mask = get_non_hydrogens()
        edge_index2 = data.edge_index[:, non_hydrogens_mask]

        if num_hydrogens != 0:
            if num_hydrogens == 1:
                subtracted_num_hydrogens = (edge_index2 > x_hydrogens).to(
                    dtype=torch.uint8
                )
            else:
                stepindex_masks = [
                    (edge_index2 > v).to(dtype=torch.uint8) for v in x_hydrogens
                ]
                subtracted_num_hydrogens = torch.stack(stepindex_masks, axis=0).sum(
                    axis=0
                )
            edge_index2 = edge_index2 - subtracted_num_hydrogens
            assert edge_index2.shape == subtracted_num_hydrogens.shape

        edge_attr2 = data.edge_attr[non_hydrogens_mask, :]

        pos2 = data.pos[x2_mask, :]

        data.x = data.x[x2_mask, :]
        data.edge_index = edge_index2
        data.edge_attr = edge_attr2
        data.pos = pos2

        return data


class AddRandomFeatures(object):
    def __init__(self):
        pass

    def __call__(self, data):
        # random_features = torch.randint(10, (data.x.size(0), 1),
        #                                 dtype=torch.float32)
        # actually adds indices!
        # random_features = torch.arange(data.x.size(0), dtype=torch.float32).unsqueeze(1)
        #         print("C", end="")
        random_features = torch.rand((data.x.size(0), 1), dtype=torch.float32)
        #         print("D", end="")
        data.x = torch.cat((data.x, random_features), dim=1)
        return data


class KGNNPrepare(object):
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms

    def __call__(self, data):
        x = data.x
        #         print("A", end="")
        #         data.x = data.x[:, :self.num_atoms]
        #         print(data.x.shape)
        #         print(data.edge_index.shape)
        #         print(data.num_nodes)
        data = TwoMalkin()(data)
        #         print("B", end="")
        data.x = x
        return data


class AddDistanceToTarget(object):
    r"""Saves the Euclidean distance of linked nodes in its edge targets."""

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data):
        (row, col), pos = data.new_edge_index, data.pos

        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        data.target_distances = dist

        return data


class RemoveBondTypes(object):
    def __init__(self):
        pass

    def __call__(self, data):
        data.edge_attr = data.edge_attr[:, 4:]
        return data


class RemoveExtraNodeFeatures:
    """
    Intended for QM9
    """

    def __init__(self, num_atoms):
        self.num_atoms = num_atoms

    def __call__(self, data):
        data.x = data.x[:, : self.num_atoms]
        return data


def build_transform(
    connect_all_atoms=False,
    remove_hydrogens=False,
    add_distance_between_atoms=False,
    add_random_features=False,
    kgnn_prepare=False,
    add_distance_as_target=False,
    add_N_hop_to_edges=False,
    hops_needed=1,
    remove_bond_types=False,
    num_atoms=11,
):
    transforms = []
    if connect_all_atoms:
        transforms.append(ConnectAll())
    if remove_hydrogens:
        transforms.append(AddSelfLoops())  # to avoid isolated nodes
        transforms.append(RemoveHydrogens())
    if add_distance_between_atoms:
        transforms.append(pyg_transforms.Distance(norm=False))
    # if remove_extra_node_features:
    #     transforms.append(RemoveExtraNodeFeatures(num_atoms))
    if add_random_features:
        transforms.append(AddRandomFeatures())
    if remove_bond_types:
        transforms.append(RemoveBondTypes())
    if kgnn_prepare:
        transforms.append(KGNNPrepare(num_atoms))
    if add_N_hop_to_edges:
        transforms.append(AddNHopToEdgeIndexAndAttr(hops_needed))
    if add_distance_as_target:
        transforms.append(AddDistanceToTarget())
    if transforms is None:
        t = None
    else:
        t = pyg_transforms.Compose(transforms)
    return t
