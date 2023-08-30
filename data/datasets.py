import re
import time
import random
import csv
import os
import warnings
import copy

import multiprocessing as mp
from multiprocessing import Process

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tqdm import tqdm
import torch
import torch_geometric
from torch_geometric.datasets import QM9


SUBDIR_SIZE = 1000  #  max num graphs in subdir

import data.Transforms as Transforms

class BaseSyntDataset(torch_geometric.data.Dataset):
    """
    Askadsky dataset (for any dataset size; so it's a bit slower)
    """


    def __init__(self, root, transform=None, pre_transform=None,
                 my_transform=None, seed_train_subset=123, indices=None,
                 nprocs=20, cyclingbefore=False, with_targets=True, target_name=None):
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
        self.graphs_dir = os.path.join(self.dir_path, 'all_graphs')
        df = pd.read_csv(self.raw_file_names)
        num_mols = len(df)
        if target_name is None:
            raise ValueError('set target (df column) name (e.g. "Tg, K")')
        self.target_name = target_name

        if self.mol_indices is None:
            self.mol_indices = torch.arange(num_mols)

        super().__init__(root, transform, pre_transform)  # calls self.process

                
        # with open(self.raw_file_names, newline='') as csvfile:
        #     self.num_mols = sum(1 for line in csvfile)
        # elif type(self.mol_indices) is int:
        graph_indices = []
        for subdir in os.listdir(self.graphs_dir):
            subdir_path = os.path.join(self.graphs_dir, subdir)
            for filename in os.listdir(subdir_path):
                graph_idx = int(filename.split('_')[1].split('.')[0])
                graph_indices.append(graph_idx)
        # print(len(graph_indices))
        # TODO add shuffle before slicing/splitting. Or it's already added there?
        self.mol_indices = torch.tensor(graph_indices, dtype=torch.int)
        if type(self.mol_indices) is int:  # ???
            self.mol_indices = self.mol_indices[:self.mol_indices]

        # self.num_mols = len(self.mol_indices)
            # self.num_mols = min(self.num_mols, self.mol_indices)
            # self.mol_indices = torch.randperm(self.num_mols)[:self.mol_indices].sort().values
        # elif type(self.mol_indices) is str:
        #     self.mol_indices = torch.load(self.mol_indices)

        self.process_targets()  # NOTE called again? not always?
        print('num mols, including bad mols:', len(self.targets))

    @property
    def raw_file_names(self):
        return os.path.join(self.raw_path, os.listdir(self.raw_path)[0])

    @property
    def processed_file_names(self):
        # Критерий необходимости препроцессинга - существование данной папки
        return ["all_graphs"]

    def download(self):
        pass

    def process_targets(self):
        print('computing mean and std of targets')
        targets = []
        df = pd.read_csv(self.raw_file_names)  # FIXME loads df again. We have two same dfs by now
        # FIXME gathers targets and measures stats, potentially INCLUDING bad (failed) mols
        targets = df[self.target_name].astype(float).to_numpy()
        # with open(self.raw_file_names, newline='') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
        #     for (_, t) in reader:
        #         targets.append(float(t))
        targets = torch.tensor(targets)
        mean, std = targets.mean(), targets.std()
        print(f'mean: {mean.item():.2f}, std: {std.item():.2f}')
        torch.save((mean, std), self.dir_path + "/meta.pt")
        self.targets = targets.view(-1,1).float()

    def process(self):
        if self.with_targets:
            self.process_targets()  # FIXME better do it AFTER self.prepare_dataset, when good mols and bad mols are known

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

    def len(self):  # ???
        return len(self.mol_indices)

    def get(self, idx):
        graph_index = self.mol_indices[idx]
        graph_subdir = torch.div(graph_index, SUBDIR_SIZE, rounding_mode="trunc")

        subdir_path = os.path.join(self.graphs_dir, f'graphs_{graph_subdir}')
        graph_path = os.path.join(subdir_path, f'graph_{graph_index}.pt')
        graph = torch.load(graph_path)
        
        graph.y = self.targets[graph_index]  # FIXME why do this? we already have graph.y, right?
        
        return graph

    def prepare_dataset(self, pid=0, nprocs=1):
        # try:
        os.makedirs(self.graphs_dir, exist_ok=True)
        # except FileExistsError:
        #     print("Directory has been created in other process.")
        #     pass
    
        db_path = self.raw_file_names

        # graph_count = pid
        mol_idx = pid
        graph_file_idx = pid
        bad_mols_idxs = []

        last_folder = -1  # -1 is to create the first graph subdir
        # endpoint_symbol = Chem.MolFromSmiles('*')
        df = pd.read_csv(db_path)#[['ID', 'SMILES', 'Tg, K']]
        # with open(db_path, newline='') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
    
        if pid==0: # Progress bar should be showed by one process
            counter = tqdm(df.iterrows(), total=len(self.mol_indices))
            # counter = tqdm(reader, total=len(self.mol_indices))
        else:
            counter = df.iterrows()
        for i, datarow in counter:
            if i != self.mol_indices[mol_idx]:  # skip all mols except every N's (nprocs')
                continue
            
            assert self.with_targets
            # try:
            (id, smiles, target) = datarow
            # except Exception as e:
                # print('len(datarow)', len(datarow))
                # for what in datarow:
                    # print('what', what)
                # print('error', datarow)
                # raise e
                
            if graph_file_idx//SUBDIR_SIZE > last_folder:
                # print(f"{pid}: Go to the next folder! Processed: {graph_file_idx}")
                last_folder = graph_file_idx//SUBDIR_SIZE
                subdir = f'graphs_{last_folder}'
                save_dir = os.path.join(self.graphs_dir, subdir)
                # try:
                os.makedirs(save_dir, exist_ok=True)
                # except FileExistsError:
                #     print(f"{pid}: Subdir is already created by other process.")
                #     pass

            try:
                graph = self.graphFromSmiles(smiles)
            except Exception:
                # something's bad about that molecule
                bad_mols_idxs.append(i)
                mol_idx += nprocs
                continue
            #except ValueError as e:
            #    print("Smth wrong", e)
            #    continue
            if self.with_targets:
                graph.y = torch.tensor(float(target), dtype=torch.float32).unsqueeze(0)
            graph.index = torch.tensor(float(i)).unsqueeze(0)
            
            assert self.my_transform is not None   # enforcing k-GNN
            graph = self.my_transform(graph)  # k-GNN
            
            save_path = os.path.join(save_dir, f'graph_{graph_file_idx}.pt')
            torch.save(graph, save_path)
            graph_file_idx += nprocs

            mol_idx += nprocs
        print(f"[pid {pid}] Processed!")
        print(f"[pid {pid}] Num bad molecules: {len(bad_mols_idxs)}")
        if bad_mols_idxs:
            print(f"[pid {pid}] Indices of bad molecules: {bad_mols_idxs}")
    
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
        try:
            edmol.AddBond(nbs[0],nbs[1],order=bond_type)
        except RuntimeError:
            # print('bond already exists, skipping...')
            pass
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
        # atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', '*')
        # NOTE when permeability pretraining (PA_syn_perm_He) only these atoms are encountered (in raw smiles):  ['C', 'N', 'O', 'F', 'S', 'I']
        atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', 'P', 'Na', '*')
        x = torch.zeros([mol.GetNumAtoms(), len(atom_symbols) + 1], dtype=torch.float32)
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            try:
                idx = atom_symbols.index(atom_symbol)
            except ValueError as e:
                print('unexpected atom:', atom_symbol)
                raise e
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
        # edge_cycle = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_types.append(bondTypes[bond.GetBondType()])
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Edge attributes
        nEdges = edge_index.shape[0]
        edgeType = torch.zeros(nEdges, nTypes)
        edgeType[torch.arange(nEdges), edge_types] = 1
        # import random
        # Dists
        # try:
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        # except ValueError as e:
        #     Draw.MolToFile(mol,f'mol_{random.randrange(0, 100000000)}.png')
        #     # print(Chem.MolToSmiles(mol))
        #     raise e
        edgeLength = torch.norm(pos[edge_index[:, 0]] - pos[edge_index[:, 1]], p=2, dim=1)
        if not self.cyclingbefore:
            edgeLength[-1] = random.gauss(mu=1.39, sigma=0.3)  # TODO test
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
        #### Change Iodine atom type to *
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
        res = AllChem.EmbedMolecule(mol)
        if res == -1:  # -1 means that embedding has failed
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res == -1:  # NOTE added
            raise RuntimeError('2D->3D molecule conversion has failed')
        t1 = time.time()
        while(AllChem.MMFFOptimizeMolecule(mol)==1):
            if time.time()-t1 > 10:  # NOTE added timeout
                break
        mol = Chem.RemoveHs(mol)
        if not self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        return self.graphFromMol(mol)

class SynteticDataset(BaseSyntDataset):
    def __init__(self, root="/storage/db/Polyimides/syntetic", seed=123,
                 indices=None, cyclingbefore=False, normalize=True,
                 with_targets=True, target_name=None):
        my_transform = Transforms.build_transform(
                                               kgnn_prepare = True,
                                               )
        transform = Transforms.build_transform(
                                               add_random_features = True,
                                               )
        super().__init__(root, my_transform=my_transform, transform=transform,
                         seed_train_subset=seed, cyclingbefore=cyclingbefore,
                         indices=indices, with_targets=with_targets, target_name=target_name)
        self.normalize = normalize
        if self.with_targets:
            # FIXME save (to load here) target stats only for good mols (excluding bad/failed)
            self.mean, self.std = torch.load(self.dir_path + "/meta.pt")

    def process(self):
        super().process()

    def get(self, idx):
        data = super().get(idx)
        if self.with_targets and self.normalize:
            data.y = (data.y - self.mean)/self.std
        return data

class ExperimentalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root="/storage/db/Polyimides/new_real", indices=None,
                 transform=None, pre_transform=None, cyclingbefore=False,
                 target_name=None):
        self.root = root
        self.mol_indices = indices
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        self.data_list = []
        if os.path.exists(self.processed_paths[0]):  # pragma: no cover
            self.data_list = torch.load(self.processed_paths[0])
        df = pd.read_csv(self.raw_file_names)
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
        self.cyclingbefore = cyclingbefore
        super().__init__(root, transform, pre_transform)
        if self.mol_indices is None:
            self.mol_indices = torch.arange(len(self.data_list))
        elif type(self.mol_indices) is int:
            self.mol_indices = torch.randperm(num_mols)[:self.mol_indices].sort().values
        elif type(self.mol_indices) is str:
            self.mol_indices = torch.load(self.mol_indices)
        self.mean, self.std, self.m_indices = torch.load(self.dir_path + "/meta.pt")

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
        indices = []
        bad_mols_count = 0
        # for polymer_id, group in tqdm(df.groupby(["ID"], as_index=False)[["ID", "SMILES", "Tg, K"]]):
            # smiles = group['SMILES'].iloc[0]
            # print(polymer_id)
            # print(smiles)
            # tg_mean = group['Tg, K'].mean()
            # tgs = group['Tg, K'].to_list()
        for idx, row in tqdm(df.iterrows()):
            smiles = row['SMILES']
            index = ''.join(filter(str.isdigit, str(row['ID'])))
            # index = ''.join(filter(str.isdigit, polymer_id))
            try:
                graph = self.graphFromSmiles(smiles)
            except Exception:
                # something's bad about that molecule
                print(f'bad molecule! ID: {row["ID"]}')
                bad_mols_count += 1
                continue
            indices.append((index, smiles))
            graph = self.my_transform(graph)
            target = torch.tensor(float(row[self.target_name]), dtype=torch.float32).unsqueeze(0)
            # target = torch.tensor(float(tg_mean), dtype=torch.float32).unsqueeze(0)
            targets.append(float(target))
            # graph.tgs = tgs
            # graph.tgs = torch.tensor(tgs, dtype=torch.float32)
            # graph.num_tgs = torch.tensor(len(tgs), dtype=torch.long).unsqueeze(0)
            # targets += tgs
            graph.index = torch.tensor(float(index)).unsqueeze(0)  # TODO change to int?
            graph.y = target
            # graph.y = tg_mean
            self.data_list.append(graph)
        print('molecules processed successfully:', len(self.data_list), 'failed:', bad_mols_count)
        targets = torch.tensor(targets)
        mean, std = targets.mean(), targets.std()
        print('mean', mean.item(), 'std', std.item())
        torch.save(self.data_list, os.path.join(self.dir_path, self.processed_file_names[0]))
        torch.save((mean, std, indices), self.dir_path + "/meta.pt")

    def __len__(self):
        return len(self.mol_indices)

    def indices(self):
        return list(range(self.__len__()))

    def get(self, idx):
        graph_index = self.mol_indices[idx]

        graph = self.data_list[graph_index]
        graph.y = (graph.y - self.mean)/self.std
        # graph.tgs = (graph.tgs - self.mean)/self.std
        return graph

    def get_index(self, idx):
        graph_index = self.mol_indices[idx]
        return self.m_indices[graph_index]



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
        # Draw.MolToFile(mol,f'mol_{random.randrange(0, 100000000)}.png')
        try:
            edmol.AddBond(nbs[0],nbs[1],order=bond_type)
        except RuntimeError:
            print('bond already exists, skipping...')
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
        # atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', '*')
        atom_symbols = ('H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'Si', 'P', 'Na', '*')
        x = torch.zeros([mol.GetNumAtoms(), len(atom_symbols) + 1], dtype=torch.float32)
        for j, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            try:
                idx = atom_symbols.index(atom_symbol)
            except ValueError as e:
                print('unexpected atom:', atom_symbol)
                raise e
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
            edgeLength[-1] = random.gauss(mu=1.39, sigma=0.3)  # TODO test
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
        #### Change Iodine atom type to *
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
        res = AllChem.EmbedMolecule(mol)
        if res == -1:  # -1 means that embedding has failed
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res == -1:
            raise RuntimeError('2D->3D molecule conversion has failed')
        t1 = time.time()
        while(AllChem.MMFFOptimizeMolecule(mol)==1):
            if time.time()-t1 > 10:
                break
        mol = Chem.RemoveHs(mol)
        if not self.cyclingbefore:
            mol = self.addCyclicConnection(mol)
        return self.graphFromMol(mol)

def split_train_val(dataset: BaseSyntDataset, test_size: int):
    print('len(dataset)', len(dataset))
    train_dataset = copy.deepcopy(dataset)  # TODO is this necessary?
    indices = train_dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]
    train_indices, test_indices = indices[:-test_size], indices[-test_size:]
    train_indices, test_indices = train_indices.sort().values, test_indices.sort().values
    test_dataset = copy.deepcopy(dataset)  # TODO is this necessary?
    train_dataset.mol_indices = train_indices
    test_dataset.mol_indices = test_indices
    return train_dataset, test_dataset

def split_subindex(dataset: BaseSyntDataset, subindex_size: int):
    train_dataset = copy.deepcopy(dataset)  # TODO is this necessary?
    indices = train_dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]  # NOTE why reshuffle again?
    test_indices = indices[-subindex_size:]
    test_indices = test_indices.sort().values
    test_dataset = copy.deepcopy(dataset)  # TODO is this necessary?
    test_dataset.mol_indices = test_indices
    return test_dataset

def k_fold_split_fixed(dataset: BaseSyntDataset, k: int):
    indices = dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]
    
    total = 0
    train_size = len(indices)//k * (k-2)
    for i in range(k):
        test_size = (len(indices)-train_size)//2
        val_size = len(indices)-train_size-test_size
        testval_size = test_size + val_size

        test_start = i*test_size
        test_end = i*test_size + test_size
        val_end = (test_end + val_size) % len(indices)

        testval = indices[i*test_size:i*test_size+testval_size]
        if i*test_size+testval_size > len(indices):
            testval_extras = i*test_size+testval_size - len(indices)
            testval = torch.cat([testval, indices[:testval_extras]])
        test_split = testval[:test_size].sort().values
        val_split = testval[test_size:].sort().values

        if val_end < test_start:
            train_before_start = val_end
            train_after_start = len(indices)
        else:
            train_before_start = 0
            train_after_start = val_end

        train_split = torch.cat([
            indices[train_before_start:test_start],
            indices[train_after_start:len(indices)]]).sort().values

        print(f'test split {i}: num: {len(test_split)}, values:\n', test_split)
        print(f'val split {i}: num: {len(val_split)}, values:\n', val_split)
        total += len(test_split)
        # print(total)
        train_dataset = copy.deepcopy(dataset)
        # train_dataset.mol_indices = torch.cat(split_i).sort().values
        train_dataset.mol_indices = train_split
        print(f'train split {i}: num: {len(train_split)}, values:\n', train_dataset.mol_indices)
        print()
        test_dataset = copy.deepcopy(dataset)
        test_dataset.mol_indices = test_split
        val_dataset = copy.deepcopy(dataset)
        val_dataset.mol_indices = val_split
        yield (train_dataset, val_dataset, test_dataset)

def k_fold_split(dataset: BaseSyntDataset, k: int):
    indices = dataset.mol_indices
    indices = indices[torch.randperm(len(indices))]
    print(len(indices))
    split_indices = list(indices.split(len(indices)//k))
    print(split_indices)
    total = 0
    for i in range(k+1):
        split_i = copy.deepcopy(split_indices)
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
