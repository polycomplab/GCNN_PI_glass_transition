import re
import time
import random
import csv
import os
import shutil
import warnings
import copy

import numpy as np
import pandas as pd
import torch
import torch_geometric
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from tqdm import tqdm

import data.Transforms as Transforms
from data.data_splitting import k_fold_split, k_fold_split_fixed
from data.KGNNDataLoader import DataLoader
from data.prediction_dataset import PredictionExperimentalDataset
from models import KGNNModel



# runtime = time.ctime()

# checkpoints_dir = "checkpoints"
checkpoints_dir = 'checkpoints/finetune/copy_finetuned_on_PI_exp_old_without_pretrain_without_fixed_split/'
# checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
# checkpoint_path = 'checkpoints/finetune/100_epochs_lr_0.0006_round_20_new_db_finetune without pretrain #Tue Sep 26 00:08:35 2023_split_0_last.pth'
seed = 12
device_index = 6


model = KGNNModel()
model.eval()
device = torch.device(f'cuda:{device_index}')
model.to(device)

# finetune.eval = False
# finetune.eval = True

old_dataset_path = "datasets/PI_exp"
old_dataset_name = old_dataset_path.split('/')[-1]

new_dataset_path = "datasets/PI_exp_new_07_03_2023"
new_dataset_name = new_dataset_path.split('/')[-1]

seed = 12

target_name = 'Tg, K'

from data.datasets import ExperimentalDataset
old_dataset = ExperimentalDataset(root=old_dataset_path, target_name=target_name)
new_dataset = ExperimentalDataset(root=new_dataset_path, target_name=target_name)

new_dataset_loader = DataLoader(
    new_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=1)

torch.manual_seed(seed)  # important for old_dataset

n_splits = 10
split_fixed = False  # old scheme for backward compatibility with old results

# finetune.eval_dataset = ExperimentalDataset(root=eval_dataset_path, target_name=finetune.target_name)

split_fn = k_fold_split_fixed if split_fixed else k_fold_split
data_splits = list(split_fn(old_dataset, n_splits))
# finetune.pretrained_weights = f"checkpoints/pretrain/{pretrains[pre_type][1]}"

new_ids_smiles = new_dataset_loader.dataset.ids_smiles
new_ids, new_smiles = zip(*new_ids_smiles)

preds = []
for checkpoint_name in sorted(os.listdir(checkpoints_dir)):
    if not checkpoint_name.endswith('.pth'):
        continue
    print('checkpoint name:', checkpoint_name)
    checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
    checkpoint = torch.load(checkpoint_path)

    parts = checkpoint_name.split('_')
    split_idx = int(parts[parts.index('split')+1])
    print('split index', split_idx)

    # mean and standard deviation of target
    mean = checkpoint['mean']
    std = checkpoint['std']

    model.load_state_dict(checkpoint['model_state_dict'])

    train_dataset, _, _ = data_splits[split_idx]
    print('len(train_dataset)', len(train_dataset))

    old_ids_smiles = train_dataset.ids_smiles
    old_ids, old_smiles = zip(*old_ids_smiles)

    # intersection = set(old_smiles).intersection(set(new_smiles))
    # print('len(intersection)', len(intersection))
    bad_count = 0
    with torch.no_grad():
        for i, data in enumerate(new_dataset_loader):
            data = data.to(device)
            mol_id = int(data.index.item())

            # find SMILES by id
            mol_smiles = None
            for id_, smiles in new_ids_smiles:
                if mol_id == int(id_):
                    mol_smiles = smiles
                    break
            if mol_smiles is None:
                assert False
                
            # TODO predict_many_v5, где будет нормализация SMILES, типа как v4, только базы наоборот вроде

            if mol_smiles in old_smiles:
            # if mol_smiles not in old_smiles:
                bad_count += 1
                continue
            
            pred = model(data).squeeze().cpu()

            pred_denormalized = (pred*std + mean).item()
            
            preds.append({
                'split': split_idx,
                'mol_id': mol_id,
                f"{target_name}, pred": pred_denormalized,
                f"{target_name}, target": data.y.squeeze().item(),
            })
    print('num unused mols:', bad_count)
preds_df = pd.DataFrame(preds)
# results_save_path = f'{checkpoints_dir}preds_on_TRAINED_subset_of_PI_exp_new_trained_on_PI_exp_old.csv'
# results_save_path = f'{checkpoints_dir}preds_on_PI_exp_new_trained_on_PI_exp_old.csv'
results_save_path = f'{checkpoints_dir}preds_on_PI_exp_new_trained_on_PI_exp_old_to_recheck.csv'
preds_df.to_csv(results_save_path, index=False)
