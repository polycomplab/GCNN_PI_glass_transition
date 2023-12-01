import copy
import os
import multiprocessing as mp
import warnings

import numpy as np
import pandas as pd
import torch
import torch_geometric
from tqdm import tqdm
from torch_geometric.datasets import QM9

from data import Transforms
from data.preprocessing import Preprocessing


SUBDIR_SIZE = 1000  #  max num graphs in subdir

    
class SynteticDataset(torch_geometric.data.Dataset):
    """Synthetic molecular dataset.
    (E.g. targets are Tg values predicted using Askadsky method.)
    it's a bit slower than InMemoryDataset version but works with
    huge datasets (100K+ molecules) that would result in
    OutOfMemory error otherwise.
    """
        
    def __init__(self, root="/storage/db/Polyimides/syntetic", seed=123,
                 indices=None, cyclingbefore=False, with_targets=True,
                 target_name=None, nprocs=20):
        my_transform = Transforms.build_transform(
                                               kgnn_prepare = True,
                                               )
        transform = Transforms.build_transform(
                                               add_random_features = True,
                                               )
        self.graphs = [] 
        self.with_targets = with_targets
        self.my_transform = my_transform
        self.root = root
        self.mol_indices = indices
        self.nprocs = nprocs
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        self.graphs_dir = os.path.join(self.dir_path, 'all_graphs')

        # assert self.mol_indices is None
        if self.mol_indices is None:
            df = pd.read_csv(self.raw_file_names)
            columns = df.columns.copy()
            num_mols = len(df)
            
            # TODO remove
            # self.real_mean = df[target_name].mean()
            # self.real_std = df[target_name].std()
            # print('self.real_mean', self.real_mean)
            # print('self.real_std', self.real_std)
            
            del df
            self.mol_indices = torch.arange(num_mols)

        if target_name is None:
            self.target_name = [c for c in columns
                                if c not in ['ID', 'SMILES']]
            # raise ValueError('set target (df column) name (e.g. "Tg, K")')
        elif not isinstance(target_name, list):
            self.target_name = [target_name]  # FIXME do the same in ExpDataset
        else:
            self.target_name = target_name
        print('targets:', self.target_name)

        self.prep = Preprocessing(cyclingbefore=cyclingbefore)
        super().__init__(root, transform, pre_transform=None)  # calls self.process
                
        # with open(self.raw_file_names, newline='') as csvfile:
        #     self.num_mols = sum(1 for line in csvfile)
        # elif type(self.mol_indices) is int:
        if isinstance(self.mol_indices, str):
            self.mol_indices = pd.read_csv(self.mol_indices)['ID'].to_list()
            self.mol_indices = torch.tensor(self.mol_indices, dtype=torch.int)
        else:
            graph_indices = []
            for subdir in os.listdir(self.graphs_dir):
                subdir_path = os.path.join(self.graphs_dir, subdir)
                for filename in os.listdir(subdir_path):
                    graph_idx = int(filename.split('_')[1].split('.')[0])
                    graph_indices.append(graph_idx)
            graph_indices = sorted(graph_indices)
            # print(len(graph_indices))
            self.mol_indices = torch.tensor(graph_indices, dtype=torch.int)
            if type(self.mol_indices) is int:  # ???
                self.mol_indices = self.mol_indices[:self.mol_indices]

        # self.num_mols = len(self.mol_indices)
            # self.num_mols = min(self.num_mols, self.mol_indices)
            # self.mol_indices = torch.randperm(self.num_mols)[:self.mol_indices].sort().values
        # elif type(self.mol_indices) is str:
        #     self.mol_indices = torch.load(self.mol_indices)

        # self.process_targets()  # NOTE called again? not always?
        # print('num mols, including bad mols:', len(self.targets))

        if self.with_targets:
            # FIXME save (to load here) target stats only for good mols (excluding bad/failed)
            self.mean, self.std = torch.load(self.dir_path + "/meta.pt")
            print('self.mean', self.mean)
            print('self.std', self.std)
            
        # self.target_idx = list(self.mean.keys()).index(target_name)  # TODO support multiple targets
        # self.target_idx = list(self.mean.keys()).index(target_name)  # TODO support multiple targets
        self.target_idxs = {tgt:i for i,tgt in enumerate(self.mean.keys())}

    @property
    def raw_file_names(self):
        return os.path.join(self.raw_path, os.listdir(self.raw_path)[0])

    @property
    def processed_file_names(self):
        # Критерий необходимости препроцессинга - существование данной папки
        return ["all_graphs"]

    def download(self):
        pass

    def process(self):
        if self.with_targets:
            self.prepare_mean_and_std()  # FIXME better do it AFTER self.prepare_dataset, when good mols and bad mols are known

        mp.set_start_method("spawn")
        procs = []
        for i in range(self.nprocs):
            p = mp.Process(target=self.prepare_dataset, args=(i, self.nprocs))
            p.start()
            procs.append(p)
        for proc in procs:
            proc.join()

    def prepare_mean_and_std(self):
        print('computing mean and std of targets')
        targets = []
        
        df = pd.read_csv(self.raw_file_names)  # FIXME loads df again. We have two same dfs by now

        permeability_unit = 'Barrer'
        if any(t_name for t_name in self.target_name
               if permeability_unit in t_name):
            
            perm_cols = [col for col in df.columns
                         if permeability_unit in col]
            if perm_cols:
                # log(x+1) is a method to normalize log-distributed targets
                # I saw it in some papers on permeability prediction
                df[perm_cols] = df[perm_cols].apply(lambda x: np.log(x+1))
            else:
                raise ValueError('Dataset does not contain the desired property')

        # FIXME gathers targets and measures stats, potentially INCLUDING bad (failed) mols
        
        targets = df[self.target_name].astype(float).to_numpy()
        # with open(self.raw_file_names, newline='') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
        #     for (_, t) in reader:
        #         targets.append(float(t))
        targets = torch.tensor(targets)
        means, stds = targets.mean(dim=0), targets.std(dim=0)
        means = {t:m for t,m in zip(self.target_name, means)}
        stds = {t:s for t,s in zip(self.target_name, stds)}
        for t_name in self.target_name:
            print(f'{t_name:>15}'
                    f': mean: {means[t_name].item():6.2f}'
                    f', std: {stds[t_name].item():5.2f}')
        torch.save((means, stds), self.dir_path + "/meta.pt")
        # self.targets = targets.view(-1,1).float()

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

        # NOTE opened by every process. Could run out of memory
        df = pd.read_csv(db_path)#[['ID', 'SMILES', 'Tg, K']]
        # with open(db_path, newline='') as csvfile:
        #     reader = csv.reader(csvfile, delimiter=',')
    
        if pid==0: # Progress bar should be showed by one process
            counter = tqdm(df.iterrows(), total=len(self.mol_indices))
            # counter = tqdm(reader, total=len(self.mol_indices))
        else:
            counter = df.iterrows()

        assert self.with_targets
        for i, datarow in counter:  # TODO replacing the loop with pandas op could make it faster?
            if mol_idx >= len(self.mol_indices):
                break  # done
            if i != self.mol_indices[mol_idx]:  # skip all mols except every N's (nprocs')
                continue
            
            try:
            # (_, _, id, smiles, target) = datarow
                # (id, smiles, target) = datarow
                smiles = datarow['SMILES']
                # mol_id = int(''.join(filter(str.isdigit, str(datarow['ID']))))  # TODO do this
                mol_id = float(''.join(filter(str.isdigit, str(datarow['ID']))))  # TODO change to int
                targets = [float(datarow[t]) for t in self.target_name]
                # target = float(datarow[self.target_name])
            # print(datarow)
            except Exception as e:
                print('len(datarow)', len(datarow))
                print('datarow items:')
                for item_idx, item in enumerate(datarow):
                    print(f'item {item_idx}: {item}')
                raise e
                
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
                graph = self.prep.graph_from_smiles(smiles)
                # graph = self.graphFromSmiles(smiles)
            except Exception as e:
                # something's bad about that molecule
                print(e)
                bad_mols_idxs.append(i)
                mol_idx += nprocs
                continue
            #except ValueError as e:
            #    print("Smth wrong", e)
            #    continue
            if self.with_targets:
                graph.y = torch.tensor(targets, dtype=torch.float32).unsqueeze(0)
            graph.index = torch.tensor(mol_id).unsqueeze(0)  # TODO change to int?
            # graph.id = torch.tensor(mol_id).unsqueeze(0)  # TODO do this instead, then preprocess all datasets again
            
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

    def __len__(self):
        return len(self.mol_indices)

    def len(self):  # inherited
        return len(self.mol_indices)

    def get(self, idx):
        graph_index = self.mol_indices[idx]
        

        graph_subdir = graph_index // SUBDIR_SIZE
        subdir_path = os.path.join(self.graphs_dir, f'graphs_{graph_subdir}')
        full_graph_path = os.path.join(subdir_path, f'graph_{graph_index}.pt')
        graph = torch.load(full_graph_path)

        # assert graph.y is None
        # if graph.y is not None:
        #     assert graph.y == self.targets[graph_index]
        # else:
        #     graph.y = self.targets[graph_index]
        return graph


class ExperimentalDataset(torch_geometric.data.InMemoryDataset):
    def __init__(self, root="/storage/db/Polyimides/new_real", indices=None,
                 transform=None, pre_transform=None, cyclingbefore=False,
                 target_name=None):
        # self.is_exp_last = (root.split('/')[-1] == 'PI_exp_new_07_03_2023')
        self.root = root
        self.mol_indices = indices
        self.dir_path = os.path.join(root, 'processed')
        self.raw_path = os.path.join(root, 'raw')
        self.data_list = []
        if os.path.exists(self.processed_paths[0]):  # pragma: no cover
            self.data_list = torch.load(self.processed_paths[0])
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
        super().__init__(root, transform, pre_transform)  # calls self.process
        if self.mol_indices is None:
            self.mol_indices = torch.arange(len(self.data_list))
        elif type(self.mol_indices) is int:
            df = pd.read_csv(self.raw_file_names)
            num_mols = len(df)
            del df
            self.mol_indices = torch.randperm(num_mols)[:self.mol_indices].sort().values
        elif type(self.mol_indices) is str:
            self.mol_indices = torch.load(self.mol_indices)
        # self.mean, self.std, self.m_indices = torch.load(self.dir_path + "/meta.pt")
        # self.mean, self.std, self.ids_smiles = torch.load(self.dir_path + "/meta.pt")
        stuff = torch.load(self.dir_path + "/meta.pt")
        self.mean, self.std, self.ids_smiles = stuff

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
        targets_for_stats = []
        ids_smiles = []
        bad_mols_count = 0
        # for polymer_id, group in tqdm(df.groupby(["ID"], as_index=False)[["ID", "SMILES", "Tg, K"]]):
            # smiles = group['SMILES'].iloc[0]
            # print(polymer_id)
            # print(smiles)
            # tg_mean = group['Tg, K'].mean()
            # tgs = group['Tg, K'].to_list()
        for idx, datarow in tqdm(df.iterrows()):
            smiles = datarow['SMILES']
            # mol_id = int(''.join(filter(str.isdigit, str(datarow['ID']))))  # TODO do this
            mol_id = float(''.join(filter(str.isdigit, str(datarow['ID']))))  # TODO change to int
            target = float(datarow[self.target_name])
            try:
                # graph = self.graphFromSmiles(smiles)
                graph = self.prep.graph_from_smiles(smiles)
            except Exception as e:
                # print(e)
                # something's bad about that molecule
                print(f'bad molecule! ID: {datarow["ID"]}')
                bad_mols_count += 1
                continue
            ids_smiles.append((mol_id, smiles))  # TODO save mol_id as int?
            graph = self.my_transform(graph)
            # target = torch.tensor(float(tg_mean), dtype=torch.float32).unsqueeze(0)
            targets_for_stats.append(target)
            # graph.tgs = tgs
            # graph.tgs = torch.tensor(tgs, dtype=torch.float32)
            # graph.num_tgs = torch.tensor(len(tgs), dtype=torch.long).unsqueeze(0)
            # targets += tgs
            # graph.index = torch.tensor(float(mol_id)).unsqueeze(0)  # TODO change to int?
            graph.index = torch.tensor(mol_id).unsqueeze(0)
            # graph.id = torch.tensor(mol_id).unsqueeze(0)  # TODO do this instead
            graph.y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
            # graph.y = tg_mean
            self.data_list.append(graph)
        print('molecules processed successfully:', len(self.data_list),
              'failed:', bad_mols_count)
        if not self.data_list:
            raise Exception('Preprocessing has failed for all molecules')
        targets_for_stats = torch.tensor(targets_for_stats)
        mean, std = targets_for_stats.mean(), targets_for_stats.std()
        print('mean', mean.item(), 'std', std.item())
        graphs_path = os.path.join(self.dir_path, self.processed_file_names[0])
        torch.save(self.data_list, graphs_path)
        torch.save((mean, std, ids_smiles), self.dir_path + "/meta.pt")

    def __len__(self):
        return len(self.mol_indices)

    def indices(self):
        return list(range(self.__len__()))

    def get(self, idx):
        graph_index = self.mol_indices[idx]

        graph = self.data_list[graph_index]
        graph = copy.copy(graph)  # to avoid duplicate transforms

        # TODO remove, this is for mixed dataset only
        # graph.is_exp_last = torch.tensor([self.is_exp_last], dtype=torch.bool)

        # graph.tgs = (graph.tgs - self.mean)/self.std
        return graph

    # def get_index(self, idx):
    #     graph_index = self.mol_indices[idx]
    #     return self.m_indices[graph_index]


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


class MixedTrainDatasetV2(torch_geometric.data.InMemoryDataset):
    def __init__(self, dataset1, dataset2):
        self.ds1 = dataset1
        self.ds2 = dataset2
        self.transform = None
        self.pretransform = None
        self.__indices__ = None
        self._indices = None

    def __len__(self):
        return len(self.ds1) + len(self.ds2)
    
    def len(self):
        return len(self.ds1) + len(self.ds2)
    
    def get(self, idx):
        if idx < len(self.ds1):
            return self.ds1.get(idx)
        else:
            return self.ds2.get(idx-len(self.ds1))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message=".*")
    dataset = SynteticDataset()
    print(dataset[0])
