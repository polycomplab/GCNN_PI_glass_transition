import copy
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric
from tqdm import tqdm

import data.Transforms as Transforms
from data.preprocessing import Preprocessing


class PredictionExperimentalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root="/storage/db/Polyimides/new_real", indices=None,
                 transform=None, pre_transform=None, cyclingbefore=False,
                 target_name=None, mean=None, std=None):
        self.root = root
        self.mol_indices = indices
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        assert mean is not None
        assert std is not None
        self.mean = mean
        self.std = std
        self.data_list = []
        if os.path.exists(self.processed_paths[0]):  # pragma: no cover
            assert False
            # self.data_list = torch.load(self.processed_paths[0])
        df = pd.read_csv(self.raw_file_names, header=None)
        num_mols = len(df)
        if target_name is None:
            raise ValueError('set target (df column) name (e.g. "Tg, K")')
        self.target_name = target_name
        # with open(self.raw_file_names, newline='') as csvfile:
        #     self.num_mols = sum(1 for line in csvfile)
        self.my_transform = Transforms.build_transform(
                                               kgnn_prepare = True,
                                               )
        transform = Transforms.build_transform(
                                               add_random_features = True,
                                               )
        self.prep = Preprocessing(cyclingbefore=cyclingbefore)
        self.cyclingbefore = cyclingbefore
        super().__init__(root, transform, pre_transform)  # calls self.process
        if self.mol_indices is None:
            self.mol_indices = torch.arange(len(self.data_list))
        elif type(self.mol_indices) is int:
            self.mol_indices = torch.randperm(num_mols)[:self.mol_indices].sort().values
        elif type(self.mol_indices) is str:
            self.mol_indices = torch.load(self.mol_indices)
        # self.mean, self.std, self.m_indices = torch.load(self.dir_path + "/meta.pt")
        self.ids_smiles = torch.load(self.dir_path + "/meta.pt")
        self.ids_smiles = {i:s for i,s in self.ids_smiles}
        # self.mean, self.std, _ = torch.load(self.dir_path + "/meta.pt")

    @property
    def raw_file_names(self):
        return os.path.join(self.raw_path, "SMILES.csv")

    @property
    def processed_file_names(self):
        return ['data_with_ind.pt']  # same but with indices added

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(self.raw_file_names, header=None)
        targets = []
        ids_smiles = []
        bad_mols_count = 0

        for row in tqdm(df.iterrows()):
            row_index, content = row 
            # NOTE using row_index as mol_id because IDs could be missing

            smiles = content[0]
            # smiles = row['SMILES']
            # index = ''.join(filter(str.isdigit, str(row['ID'])))
            # index = ''.join(filter(str.isdigit, polymer_id))
            try:
                graph = self.prep.graph_from_smiles(smiles)
            except Exception:
                # something's bad about that molecule
                print(f'bad molecule! smiles: {smiles}')
                bad_mols_count += 1
                continue
            ids_smiles.append((row_index, smiles))
            graph = self.my_transform(graph)

            graph.index = torch.tensor(int(row_index)).unsqueeze(0)
            # graph.id = torch.tensor(int(row_index)).unsqueeze(0) # TODO

            self.data_list.append(graph)
        print('molecules processed successfully:', len(self.data_list), 'failed:', bad_mols_count)
        targets = torch.tensor(targets)
        torch.save(ids_smiles, self.dir_path + "/meta.pt")

    def __len__(self):
        return len(self.mol_indices)

    def indices(self):
        return list(range(self.__len__()))

    def get(self, idx):
        graph_index = self.mol_indices[idx]

        graph = self.data_list[graph_index]
        graph = copy.copy(graph)  # to avoid duplicate transforms

        # graph.tgs = (graph.tgs - self.mean)/self.std
        return graph
