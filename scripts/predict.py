import os
import shutil

import pandas as pd
import torch

from data.KGNNDataLoader import DataLoader
from data.prediction_dataset import PredictionExperimentalDataset
from models import KGNNModel


checkpoint_path = ('checkpoints/finetune/100_epochs_lr_0.0006_round_20_new_db_'
                   'finetune without pretrain #Tue Sep 26 00:08:35 2023_split_0_last.pth')

model = KGNNModel()
checkpoint = torch.load(checkpoint_path)

# mean and standard deviation of target property values
mean = checkpoint['mean']
std = checkpoint['std']

model.load_state_dict(checkpoint['model_state_dict'])


predictions_path = 'datasets/prediction/'
smiles_path = predictions_path + 'SMILES.csv'
temp_dataset_path = predictions_path + 'temp/'
dataset_path_raw = temp_dataset_path+'raw/'
dataset_path_processed = temp_dataset_path+'processed/'
os.makedirs(dataset_path_raw, exist_ok=True)
os.makedirs(dataset_path_processed, exist_ok=True)
shutil.copy(smiles_path, dataset_path_raw)

target_name = 'Tg, K'
dataset = PredictionExperimentalDataset(
    root=temp_dataset_path,
    target_name=target_name,
    mean=mean,
    std=std)
test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1)

model.eval()
data_format = {"SMILES": [], f"{target_name}, pred": []}
preds_result_df = pd.DataFrame(data_format)
with torch.no_grad():
    for i, test_data in enumerate(test_loader):
        pred = model(test_data).squeeze()
        smiles = dataset.ids_smiles[test_data.index.item()]
        pred_denormalized = (pred*std + mean).cpu().numpy().tolist()
        data_batch = {
            "SMILES": [smiles],
            f"{target_name}, pred": [pred_denormalized]}
        print(data_batch)
        preds_result_df = preds_result_df.append(
            pd.DataFrame(data_batch), ignore_index=True)
                
results_save_path = f'{predictions_path}/predictions.csv'
preds_result_df.to_csv(results_save_path, index=False)

shutil.rmtree(temp_dataset_path)
