import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score
from tqdm import tqdm

from radam import RAdam
from data.KGNNDataLoader import DataLoader
from data.data_splitting import split_train_subset, split_train_val, split_subindex
from utils import load_pretrained_weights


def eval_simple_on_subset_multitarget(config):
    t1 = time.time()
    torch.manual_seed(config.seed)

    dataset = config.finetune.dataset
    # if config.finetune.eval:
    #     eval_dataset = config.finetune.eval_dataset
    #     eval_dataset.std = dataset.std
    #     eval_dataset.mean = dataset.mean
    # else:
    #     eval_dataset = None

    # print(f'current dataset mean {dataset.mean:.2f} and std {dataset.std:.2f}')

    train_targets = config.finetune.train_targets  # target chemical properties
    train_targets_split = train_targets.split('_pretrained_on_')
    # if len(train_targets_split) == 1:
    #     cp_load_path = None
    # else:
    cp_load_prefix = os.path.join(config.checkpoints_dir,
                                  config.finetune.train_targets)
    cp_load_path = f'{cp_load_prefix}_last.pth'
    # pretrain_targets = train_targets_split[1].split('_and_')

    train_targets = train_targets_split[0].split('_and_')


    save_preds = config.finetune.save_preds
    # if save_preds:
    #     data_format = {"ID": [], f"{target_name}, pred": []}
    #     results_table = pd.DataFrame(data_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    # eval_tables = []
    _, test_dataset = \
        split_train_val(dataset, test_size=config.finetune.test_subset_size)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=700)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=750)
    # if (hasattr(config.finetune, 'subset_size')
    #     and config.finetune.subset_size is not None):
    #     train_dataset = split_subindex(train_dataset, config.finetune.subset_size)
    
    # print(f'train_dataset size:', len(train_dataset))
    print(f'test_dataset size:', len(test_dataset))

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=2,  # FIXME this is only for the huge dataset to prevent OOM
    #     # num_workers=4,
    #     drop_last=True)  # NOTE drop_last

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4)

    # log_path = os.path.join(config.checkpoints_dir, 'tb_logs')
    # log_path = os.path.join(log_path, config.finetune.train_targets)
    # os.makedirs(log_path, exist_ok=True)

    # log_path = config.log_dir + "/finetune/" + config.name +\
    #     f'_simple_subset_{len(train_dataset)}'

    cp_names_to_targets = {
        "Tg": ["Tg, K"],
        "perm_He": ["He, Barrer"],
        "perm_CH4": ["CH4, Barrer"],
        "perm_CO2": ["CO2, Barrer"],
        "perm_N2": ["N2, Barrer"],
        "perm_O2": ["O2, Barrer"],
        "perm_all": [
            "He, Barrer",
            "CH4, Barrer",
            "CO2, Barrer",
            "N2, Barrer",
            "O2, Barrer",
            ]
    }
    target_names = []
    for sub_targets in train_targets:
        target_names += cp_names_to_targets[sub_targets]
    assert target_names
    # target_names = cp_names_to_targets[train_targets]
    print('starting:', config.finetune.train_targets,
          'on device:', config.device_index)

    # if cp_load_path is not None:
    #     pretrain_target_names = []
    #     for sub_pretrain_targets in pretrain_targets:
    #         pretrain_target_names += cp_names_to_targets[sub_pretrain_targets]
    #     pretrain_target_idxs = [idx for tgt,idx
    #                  in train_loader.dataset.target_idxs.items()
    #                  if tgt in pretrain_target_names]
        

    model = config.model
    # dataset_mean = dataset.mean
    # dataset_std = dataset.std
    dataset_means = dataset.mean#[target_name]
    dataset_stds = dataset.std#[target_name]
    
    # TODO remove
    # dataset_mean = dataset.real_mean
    # dataset_std = dataset.real_std

    if cp_load_path is not None:
        loaded_dict = torch.load(
            cp_load_path,
            map_location=torch.device(config.device_index))
        # if 'model_state_dict' in state_dict:
        state_dict = loaded_dict['model_state_dict']

        # sd_new = {}
        # for p_name, p in state_dict.items():
        #     if p_name.startswith('heads'):
        #         head_idx = int(p_name.split('.')[1])
                
        #         # avoiding copying targets if didn't train on them
        #         # This is for fixing failed finetuning
        #         # because of zero weights for other heads
        #         #  that happened due to weight decay
        #         if head_idx not in pretrain_target_idxs:
        #             continue
        #     sd_new[p_name] = p
        # state_dict = sd_new
        del loaded_dict
    else:
        state_dict = model.state_dict()

    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    model.load_state_dict(state_dict, strict=False)
    # model.train()

    cp_save_prefix = os.path.join(config.checkpoints_dir,
                                  f'{config.finetune.train_targets}')
    # cp_save_path = f'{cp_save_prefix}_last.pth'
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune.epochs, eta_min=1e-6)
    # new_results_table, new_eval_table = train_on_split(
    # save_frequency = 1  # in epochs

    train_targets = {tgt:idx for tgt,idx
                     in test_loader.dataset.target_idxs.items()
                     if tgt in target_names}
    # print('train_targets', train_targets)
    train_target_idxs_mask = torch.tensor(
        list(train_targets.values()), device=device)
    assert train_target_idxs_mask.equal(train_target_idxs_mask.sort()[0])
    # print('train_target_idxs_mask', train_target_idxs_mask)

    dataset_means_tensor = torch.tensor(
        [dataset_means[tgt] for tgt in train_targets.keys()],
         device=device).unsqueeze(0)
    # print('dataset_means_tensor', dataset_means_tensor)
    dataset_stds_tensor = torch.tensor(
        [dataset_stds[tgt] for tgt in train_targets.keys()],
         device=device).unsqueeze(0)
    
    train_perm_targets = [new_idx for new_idx,tgt
                          in enumerate(train_targets.keys())
                          if 'Barrer' in tgt]
    # print('train_perm_targets', train_perm_targets)
    if train_perm_targets:
        train_perm_targets_mask = torch.tensor(
            train_perm_targets,
            device=device)
    else:
        train_perm_targets_mask = None

    test_loss, test_MAE, test_RMSE, test_R2, preds_df = evaluate(
            test_loader=test_loader,
            save_preds=save_preds,
            # target_name=target_name,
            target_names=target_names,
            model=model,
            dataset_means=dataset_means,
            dataset_stds=dataset_stds,
            device=device,
            config=config,
            train_target_idxs_mask=train_target_idxs_mask,
            train_perm_targets=train_perm_targets,
            train_perm_targets_mask=train_perm_targets_mask,
            dataset_means_tensor=dataset_means_tensor,
            dataset_stds_tensor=dataset_stds_tensor,
             )

    csv_dict = {}
    # 'test_loss': test_loss
    for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        csv_dict[f'Test_Loss_MSE_{tgt_name}'] = [float(test_loss[tgt_idx])]
    for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        csv_dict[f'Test_RMSE_{tgt_name}'] = [float(test_RMSE[tgt_idx])]
    for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        csv_dict[f'Test_R2_{tgt_name}'] = [float(test_R2[tgt_idx])]
    for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        csv_dict[f'Test_MAE_{tgt_name}'] = [float(test_MAE[tgt_idx])]
    csv_df = pd.DataFrame(csv_dict)
    save_path = f'{cp_save_prefix}_last_logperm.csv'
    csv_df.to_csv(save_path, index=False)


    t2 = time.time()
    training_time_in_hours = (t2-t1)/60/60
    print(f'"{config.finetune.train_targets}" finished.'
          f' Training time: {training_time_in_hours:.2f} hours.', flush=True)


def evaluate(*, test_loader, save_preds, target_names, model,
             dataset_means, dataset_stds, device, config,
             train_target_idxs_mask,
             train_perm_targets, train_perm_targets_mask,
             dataset_means_tensor, dataset_stds_tensor):
    model.eval()
    # test_loss = 0.
    # test_MAE = 0.
    test_loss = torch.zeros(len(target_names), device=device)
    test_RMSE = torch.zeros(len(target_names), device=device)
    test_MAE = torch.zeros(len(target_names), device=device)
    # val_progressbar = tqdm(test_loader, desc=f"Validation")
    # if save_preds:
    #     data_format = {"ID": [], f"{target_name}, pred": []}
    #     preds_df = pd.DataFrame(data_format)
    # else:
    preds_df = None
    # data_mean = dataset_mean.item()
    # data_std = dataset_std.item()
    all_preds = []
    all_targets = []
    num_mols = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_data = test_data.to(device)
            pred = model(test_data).squeeze()
            # if i == len(test_loader)-1:
            #     print('test')
            #     print('pred.shape', pred.shape)
            #     print('pred:')
            #     print(pred)
            pred = pred[:, train_target_idxs_mask]
            
            # if i == len(test_loader)-1:
            #     print('filtered pred:')
            #     print('pred.shape', pred.shape)
            #     print('pred:')
            #     print(pred)
            
            # if save_preds:
            #     mol_ids = test_data.index.cpu().numpy().tolist()
            #     pred_denormalized = (pred*data_std + data_mean).cpu().numpy().tolist()
            #     data_batch = {
            #         "ID": mol_ids,
            #         f"{target_name}, pred": pred_denormalized}  # FIXME add log for perms
            #     # print(data_batch)
            #     preds_df = preds_df.append(  # TODO improve speed here
            #         pd.DataFrame(data_batch), ignore_index=True)
            
            # range_mask = (test_data.y>=config.min_t)&(test_data.y<=config.max_t)  # NOTE only for Tg
            # pred = pred[range_mask]
            num_mols += len(pred)
            # target = test_data.y[:, ]

            pred_denormalized = pred*dataset_stds_tensor+dataset_means_tensor
            # if i == len(test_loader)-1:
            #     print('pred_denormalized:')
            #     print(pred_denormalized)

            # NOTE not log-denormalizing permeability targets anymore
            # if train_perm_targets:
            #     pred_denormalized[:, train_perm_targets_mask] =\
            #         torch.exp(pred_denormalized[:, train_perm_targets_mask])-1

            # if i == len(test_loader)-1:
            #     print('pred_denormalized after train_perm_targets:')
            #     print(pred_denormalized)
                
            target_original = test_data.y[:, train_target_idxs_mask]
            target_original = target_original.float()  # double -> float
            # if i == len(test_loader)-1:
            #     print('target_original:')
            #     print(target_original)

            log_target = target_original.clone()
            if train_perm_targets:
                log_target[:, train_perm_targets_mask] =\
                      torch.log(log_target[:, train_perm_targets_mask]+1)
            # target = torch.log(target+1)
            target = log_target.clone()
            target = (target - dataset_means_tensor)/dataset_stds_tensor
            target = target.float()  # double -> float
            # if i == len(test_loader)-1:
            #     print('target:')
            #     print(target)
            # pred_denormalized = pred*dataset_std+dataset_mean
            # pred_denormalized = torch.exp(pred_denormalized)-1
            
            # target_original = test_data.y[:, test_loader.dataset.target_idx]
            # target = (target_original - dataset_mean)/dataset_std
            # target = torch.log(target+1)
            # target = torch.log(target+1)
            # target_normalized = (target - data_mean)/data_std
            # target_normalized = target_normalized[range_mask]
            # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
            # test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()  # TODO uncomment
            # test_loss += F.mse_loss(pred, target, reduction='sum').item()
            test_loss += F.mse_loss(pred, target, reduction='none').sum(dim=0)
            # test_RMSE += F.mse_loss(pred_denormalized, target_original, reduction='none').sum(dim=0)
            test_RMSE += F.mse_loss(pred_denormalized, log_target, reduction='none').sum(dim=0)
            # test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()  # TODO uncomment
            # test_MAE += F.l1_loss(pred_denormalized, target_original, reduction='sum').item()
            # test_MAE += F.l1_loss(pred_denormalized, target_original, reduction='none').sum(dim=0)
            test_MAE += F.l1_loss(pred_denormalized, log_target, reduction='none').sum(dim=0)

            all_preds.append(pred_denormalized.cpu().numpy())
            # all_targets.append(target_original.cpu().numpy())
            all_targets.append(log_target.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    test_R2 = r2_score(all_targets, all_preds, multioutput='raw_values')

    # test_loss /= len(test_loader.dataset)
    test_loss /= num_mols
    test_RMSE = torch.sqrt(test_RMSE/num_mols)
    # test_MAE /= len(test_loader.dataset)
    test_MAE /= num_mols
    # test_MAE_denormalized = test_MAE*data_std
    # print(f"Test loss: {test_loss:.4f}, test MAE (Tg, K): {test_MAE_denormalized:.4f}")
    return test_loss, test_MAE, test_RMSE, test_R2, preds_df

