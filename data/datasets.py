import re
import time
import random
import csv
import pandas as pd
import os
import warnings
import copy

import multiprocessing as mp

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import torch_geometric
from torch_geometric.datasets import QM9

from tqdm import tqdm
from multiprocessing import Process

SUBDIR_SIZE = 1000  #  max num graphs in subdir

import data.Transforms as Transforms

class BaseSyntDataset(torch_geometric.data.Dataset):
    """
    Askadsky dataset (for any dataset size; so it's a bit slower)
    """


    def __init__(self, root, transform=None, pre_transform=None, my_transform=None, seed_train_subset=123,
                 indices=None, nprocs=20, cyclingbefore=False, with_targets=True):
        """
        seed - provide the same seed both for the train dataset (is_training=True) 
        and test dataset (is_training=False)
        """
        self.graphs = [] 
        self.with_targets = with_targets

        self.my_transform = my_transform
        self.cyclingbefore = cyclingbefore
        self.root = root
        self.mol_indices = indices
        self.nprocs = nprocs
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        with open(self.raw_file_names, newline='') as csvfile:
            self.num_mols = sum(1 for line in csvfile)
        if self.mol_indices is None:
            print("Here")
            self.mol_indices = torch.arange(self.num_mols)
        elif type(self.mol_indices) is int:
            self.mol_indices = torch.randperm(self.num_mols)[:self.mol_indices].sort().values
        elif type(self.mol_indices) is str:
            self.mol_indices = torch.load(self.mol_indices)

        self.graphs_dir = os.path.join(self.dir_path, 'all_graphs')
        self.process_targets()
        print(len(self.targets))

        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return os.path.join(self.raw_path, os.listdir(self.raw_path)[0])

    @property
    def processed_file_names(self):
        return ["all_graphs"]

    def download(self):
        pass

    def process_targets(self):
        targets = []
        with open(self.raw_file_names, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for (_, t) in reader:
                targets.append(float(t))
            targets = torch.tensor(targets)
            mean, std = targets.mean(), targets.std()
            torch.save((mean, std), self.dir_path + "/meta.pt")
        self.targets = targets.view(-1,1)

    def process(self):
        if self.with_targets:
            self.process_targets()

        mp.set_start_method("spawn")
        procs = []
        for i in range(self.nprocs):
            p = Process(target=self.prepare_dataset, args=(i, self.nprocs))
            p.start()
            procs.append(p)
        for proc in procs:
            proc.join()

    def __len__(self):
        return len(self.mol_indices)

    def len(self):
        return len(self.mol_indices)

    def get(self, idx):
        graph_index = self.mol_indices[idx]
        graph_subdir = torch.div(graph_index, SUBDIR_SIZE, rounding_mode="trunc")

        subdir_path = os.path.join(self.graphs_dir, f'graphs_{graph_subdir}')
        graph_path = os.path.join(subdir_path, f'graph_{graph_index}.pt')
        graph = torch.load(graph_path)
        graph.y = self.targets[graph_index]
        return graph

    def prepare_dataset(self, pid=0, nprocs=1):
        try:
            os.makedirs(self.graphs_dir)
        except FileExistsError:
            print("Directory has been created in other process.")
            pass
    
        db_path = self.raw_file_names

        graph_count = pid
        mol_idx = pid
        batch_count = pid
        graphs = []

        last_folder = -1
        endpoint_symbol = Chem.MolFromSmiles('*')
        with open(db_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
    
            if pid==0: # Progress bar should be showed by one process
                counter = tqdm(reader, total=len(self.mol_indices))
            else:
                counter = reader
            for i, datarow in enumerate(counter):
                if i != self.mol_indices[mol_idx]:
                    continue
                if self.with_targets:
                    (smiles, target) = datarow
                else:
                    smiles = datarow[0]
                    
                if batch_count//SUBDIR_SIZE > last_folder:
                    print(f"{pid}: Go to the next folder! Processed: {batch_count}")
                    last_folder=batch_count//SUBDIR_SIZE
                    subdir = f'graphs_{last_folder}'
                    save_dir = os.path.join(self.graphs_dir, subdir)
                    try:
                        os.makedirs(save_dir)
                    except FileExistsError:
                        print(f"{pid}: Subdir is already created by other process.")
                        pass

                graph = self.graphFromSmiles(smiles)
                #except ValueError as e:
                #    print("Smth wrong", e)
                #    continue
                if self.with_targets:
                    target_tg = torch.tensor(float(target), dtype=torch.float32).unsqueeze(0)
                    graph.y = target_tg
                graph.index = torch.tensor(float(i)).unsqueeze(0)
                if self.my_transform is not None:
                    graph = self.my_transform(graph)
                graphs.append(graph)
                if(len(graphs)==1):
                    save_path = os.path.join(save_dir, f'graph_{batch_count}.pt')
                    torch.save(graphs[0], save_path)
                    batch_count+=nprocs
                    graphs = []

                mol_idx += nprocs
            print("Processed!")
    
    @staticmethod        
    def addCyclicConnection(mol):
        stars = []
        nbs = []
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                bonds = list(atom.GetBonds())
                assert len(bonds) == 1
                stars.append(atom.GetIdx())
                bond_type = bonds[0].GetBondType()
                for a in atom.GetNeighbors():
                    nbs.append(a.GetIdx())
        edmol = Chem.EditableMol(mol)
        edmol.AddBond(nbs[0],nbs[1],order=bond_type)
        if (stars[0]>stars[1]):
            edmol.RemoveAtom(stars[0])
            edmol.RemoveAtom(stars[1])
        else:
            edmol.RemoveAtom(stars[1])
            edmol.RemoveAtom(stars[0])
        return edmol.GetMol()
    
#     @staticmethod
    def graphFromMol(self, mol):
        #### Vertices data
        atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', '*')
        x = torch.zeros([mol.GetNumAtoms(), len(atom_symbols) + 1], dtype=torch.float32)
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            idx = atom_symbols.index(atom_symbol)
            x[j, idx] = 1
            x[j, -1] = atom.GetExplicitValence() - atom.GetDegree()
        #### Edge data
        nTypes = 4
        bondTypes = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
        edge_index = []
        edge_types = []
        edge_cycle = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_types.append(bondTypes[bond.GetBondType()])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Edge attributes
        nEdges = edge_index.shape[0]
        edgeType = torch.zeros(nEdges, nTypes)
        edgeType[torch.arange(nEdges), edge_types] = 1
        # Dists
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        edgeLength = torch.norm(pos[edge_index[:, 0]] - pos[edge_index[:, 1]], p=2, dim=1)
        if not self.cyclingbefore:
            edgeLength[-1] = random.gauss(mu=1.39, sigma=0.3)
        # Cyclic info
        edgeCyc = torch.zeros(nEdges, 5)
        for i, bond in enumerate(mol.GetBonds()):
            for k in range(4, 9):
                if bond.IsInRingSize(k):
                    edgeCyc[k-4] = 1
        edgeAttrs = torch.cat([edgeType, edgeLength.unsqueeze(-1), edgeCyc], dim = 1)
        edgeAttrs = edgeAttrs.repeat(2, 1)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).contiguous()
        return torch_geometric.data.Data(x=x, edge_index=edge_index.t(), edge_attr=edgeAttrs, pos=pos)
    

    def graphFromSmiles(self, smiles):
        #### Change Iod to *
        smiles = list(smiles)
        for i, l in enumerate(smiles):
            if(l=="I"):
                smiles[i]="*"
        smiles = "".join(smiles)
        #####
        mol = Chem.MolFromSmiles(smiles)
        
        if self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        
        #### Embedding
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        while(AllChem.MMFFOptimizeMolecule(mol)==1): pass
        mol = Chem.RemoveHs(mol)
        if not self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        return self.graphFromMol(mol)

class SynteticDataset(BaseSyntDataset):
    def __init__(self, root="/storage/db/Polyimides/syntetic", seed=123, indices=None, cyclingbefore=False, normalize=True, with_targets=True):
        my_transform = Transforms.build_transform(
                                               kgnn_prepare = True,
                                               )
        transform = Transforms.build_transform(
                                               add_random_features = True,
                                               )
        super().__init__(root, my_transform=my_transform, transform=transform, seed_train_subset=seed, cyclingbefore=cyclingbefore, indices=indices, with_targets=with_targets)
        self.normalize = normalize
        if self.with_targets:
            self.mean, self.std = torch.load(self.dir_path + "/meta.pt")

    def process(self):
        super().process()

    def get(self, idx):
        data = super().get(idx)
        if self.with_targets and self.normalize:
            data.y = (data.y - self.mean)/self.std
        return data

class ExperimentalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root="/storage/db/Polyimides/new_real", indices=None, transform=None, pre_transform=None, cyclingbefore=False):
        self.root = root
        self.mol_indices = indices
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        self.data_list = []
        if os.path.exists(self.processed_paths[0]):  # pragma: no cover
            self.data_list = torch.load(self.processed_paths[0])
        with open(self.raw_file_names, newline='') as csvfile:
            self.num_mols = sum(1 for line in csvfile)
        self.my_transform = Transforms.build_transform(
                                               kgnn_prepare = True,
                                               )
        transform = Transforms.build_transform(
                                               add_random_features = True,
                                               )
        self.cyclingbefore = cyclingbefore
        super().__init__(root, transform, pre_transform)
        if self.mol_indices is None:
            self.mol_indices = torch.arange(len(self.data_list))
        elif type(self.mol_indices) is int:
            self.mol_indices = torch.randperm(self.num_mols)[:self.mol_indices].sort().values
        elif type(self.mol_indices) is str:
            self.mol_indices = torch.load(self.mol_indices)
        self.mean, self.std, self.indeces = torch.load(self.dir_path + "/meta.pt")

    @property
    def raw_file_names(self):
        return os.path.join(self.raw_path, "target_csv.csv")

    @property
    def processed_file_names(self):
        return ['data_with_ind.pt']  # same but with indices added

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(self.raw_file_names)
        targets = []
        indeces = []
        for idx, row in df.iterrows():
            smiles = row['SMILES']
            index = row['ID']
            indeces.append((index, smiles))
            graph = self.graphFromSmiles(smiles)
            graph = self.my_transform(graph)
            target = torch.tensor(float(row['Tg, K']), dtype=torch.float32).unsqueeze(0)
            targets.append(float(target))
            graph.index = torch.tensor(float(index)).unsqueeze(0)
            graph.y = target
            self.data_list.append(graph)
        targets = torch.tensor(targets)
        mean, std = targets.mean(), targets.std()
        torch.save(self.data_list, os.path.join(self.dir_path, self.processed_file_names[0]))
        torch.save((mean, std, indeces), self.dir_path + "/meta.pt")

    def __len__(self):
        return len(self.mol_indices)

    def indices(self):
        return list(range(self.__len__()))

    def get(self, idx):
        graph_index = self.mol_indices[idx]

        graph = self.data_list[graph_index]
        graph.y = (graph.y - self.mean)/self.std
        return graph

    def get_index(self, idx):
        graph_index = self.mol_indices[idx]
        return self.indeces[graph_index]



    @staticmethod        
    def addCyclicConnection(mol):
        stars = []
        nbs = []
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            if atom_symbol == '*':
                bonds = list(atom.GetBonds())
                assert len(bonds) == 1
                stars.append(atom.GetIdx())
                bond_type = bonds[0].GetBondType()
                for a in atom.GetNeighbors():
                    nbs.append(a.GetIdx())
        edmol = Chem.EditableMol(mol)
        edmol.AddBond(nbs[0],nbs[1],order=bond_type)
        if (stars[0]>stars[1]):
            edmol.RemoveAtom(stars[0])
            edmol.RemoveAtom(stars[1])
        else:
            edmol.RemoveAtom(stars[1])
            edmol.RemoveAtom(stars[0])
        return edmol.GetMol()
    
#     @staticmethod
    def graphFromMol(self, mol):
        #### Vertices data
        atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', '*')
        x = torch.zeros([mol.GetNumAtoms(), len(atom_symbols) + 1], dtype=torch.float32)
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            idx = atom_symbols.index(atom_symbol)
            x[j, idx] = 1
            x[j, -1] = atom.GetExplicitValence() - atom.GetDegree()
        #### Edge data
        nTypes = 4
        bondTypes = {
                Chem.rdchem.BondType.SINGLE: 0,
                Chem.rdchem.BondType.DOUBLE: 1,
                Chem.rdchem.BondType.TRIPLE: 2,
                Chem.rdchem.BondType.AROMATIC: 3,
            }
        edge_index = []
        edge_types = []
        edge_cycle = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_types.append(bondTypes[bond.GetBondType()])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Edge attributes
        nEdges = edge_index.shape[0]
        edgeType = torch.zeros(nEdges, nTypes)
        edgeType[torch.arange(nEdges), edge_types] = 1
        # Dists
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        edgeLength = torch.norm(pos[edge_index[:, 0]] - pos[edge_index[:, 1]], p=2, dim=1)
        if not self.cyclingbefore:
            edgeLength[-1] = random.gauss(mu=1.39, sigma=0.3)
        # Cyclic info
        edgeCyc = torch.zeros(nEdges, 5)
        for i, bond in enumerate(mol.GetBonds()):
            for k in range(4, 9):
                if bond.IsInRingSize(k):
                    edgeCyc[k-4] = 1
        edgeAttrs = torch.cat([edgeType, edgeLength.unsqueeze(-1), edgeCyc], dim = 1)
        edgeAttrs = edgeAttrs.repeat(2, 1)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).contiguous()
        return torch_geometric.data.Data(x=x, edge_index=edge_index.t(), edge_attr=edgeAttrs, pos=pos)
    

    def graphFromSmiles(self, smiles):
        #### Change Iod to *
        smiles = list(smiles)
        for i, l in enumerate(smiles):
            if(l=="I"):
                smiles[i]="*"
        smiles = "".join(smiles)
        #####
        mol = Chem.MolFromSmiles(smiles)
        
        if self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        
        #### Embedding
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        while(AllChem.MMFFOptimizeMolecule(mol)==1): pass
        mol = Chem.RemoveHs(mol)
        if not self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        return self.graphFromMol(mol)

def split_train_val(dataset: BaseSyntDataset, test_size: int):
    print(len(dataset))
    train_dataset = copy.deepcopy(dataset)
    indeces = train_dataset.mol_indices
    indeces = indeces[torch.randperm(len(indeces))]
    train_indeces, test_indeces = indeces[:-test_size], indeces[-test_size:]
    train_indeces, test_indeces = train_indeces.sort().values, test_indeces.sort().values
    test_dataset = copy.deepcopy(dataset)
    train_dataset.mol_indices = train_indeces
    test_dataset.mol_indices = test_indeces
    return train_dataset, test_dataset

def split_subindex(dataset: BaseSyntDataset, subindex_size: int):
    train_dataset = copy.deepcopy(dataset)
    indeces = train_dataset.mol_indices
    indeces = indeces[torch.randperm(len(indeces))]
    test_indeces = indeces[-subindex_size:]
    test_indeces = test_indeces.sort().values
    test_dataset = copy.deepcopy(dataset)
    test_dataset.mol_indices = test_indeces
    return test_dataset

def k_fold_split(dataset: BaseSyntDataset, k: int):
    indeces = dataset.mol_indices
    indeces = indeces[torch.randperm(len(indeces))]
    print(len(indeces))
    split_indeces = list(indeces.split(len(indeces)//k))
    print(split_indeces)
    total = 0
    for i in range(k+1):
        split_i = copy.deepcopy(split_indeces)
        test_split = split_i.pop(i).sort().values
        val_split = split_i.pop((i)%(k)).sort().values
        total += len(test_split)
        print(total)
        train_dataset = copy.deepcopy(dataset)
        train_dataset.mol_indices = torch.cat(split_i).sort().values
        test_dataset = copy.deepcopy(dataset)
        test_dataset.mol_indices = test_split
        val_dataset = copy.deepcopy(dataset)
        val_dataset.mol_indices = val_split
        yield (train_dataset, val_dataset, test_dataset)

class QM9Dataset(QM9):
    def __init__(self):
        self.my_transform = Transforms.build_transform(
                                               add_random_features = True,
                                               kgnn_prepare = True,
                                               )
        super().__init__(root="/storage/3050/db/Polyimides/QM9", pre_transform=self.my_transform)
        delattr(self.data, "name")
        self.mol_indices = torch.arange(0,super().len())
        self.std, self.mean = torch.std_mean(self.data.y[:, 1])

    def download(self):
        super().download()

    def process(self):
        super().process()

    def __len__(self):
        return len(self.mol_indices)

    def get(self, idx):
        data = super().get(self.mol_indices[idx])
        data.y = data.y[:, 3]
        return data

class MixedSynteticDataset(torch_geometric.data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.ds1 = dataset1
        self.ds2 = dataset2
        self.transform = None
        self.pretransform = None
        self.mol_indices  = torch.arange(0, len(self.ds1) + len(self.ds2))
        self.__indices__ = None
        self._indices = None
        self.normalize()


    def __len__(self):
        return len(self.mol_indices)
    def len(self):
        return len(self.mol_indices)
    def get(self, idx):
        idx = self.mol_indices[idx]
        if idx < len(self.ds1):
            return self.ds1.get(idx)
        else:
            return self.ds2.get(idx-len(self.ds1))

    def normalize(self):
        try:
            self.mean, self.std = torch.load("mixed_data_norm.pt")
        except FileNotFoundError:
            denorm_targets = []
            for i in range(len(self.ds1)):
                denorm_targets.append(self.ds1.get(i).y*self.ds1.std + self.ds1.mean)
            for i in range(len(self.ds2)):
                denorm_targets.append(self.ds2.get(i).y*self.ds2.std + self.ds2.mean)
            denorm_targets = torch.stack(denorm_targets)
            self.mean, self.std = denorm_targets.mean(), denorm_targets.std()
            torch.save((self.mean, self.std), "mixed_data_norm.pt")
        self.ds1.mean, self.ds1.std = self.mean, self.std
        self.ds2.mean, self.ds2.std = self.mean, self.std



if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*")
    dataset = SynteticDataset()
    print(dataset[0])
