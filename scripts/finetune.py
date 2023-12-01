import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from radam import RAdam
from data.KGNNDataLoader import DataLoader
from data.data_splitting import k_fold_split, k_fold_split_fixed
from utils import load_pretrained_weights


def update_finetune_config(args, config):
    # Explicitly provided args are useful for
    # training multiple finetunes in sequence

    args_dict = vars(args)
    if args_dict["device_idx"] is not None:
        config.device_index = args_dict["device_idx"]
    if args_dict["epochs"] is not None:
        config.finetune.epochs = args_dict["epochs"]
    if args_dict["lr"] is not None:
        config.finetune.optimizer_params["lr"] = args_dict["lr"]
    if args_dict["loss2_weight"] is not None:
        config.finetune.loss2_weight = args_dict["loss2_weight"]
    # if args_dict["num_task_layers"] is not None:  # TODO implement?
    #     config.num_task_layers = args_dict["num_task_layers"]
    if args_dict["subset_size"] is not None:
        config.finetune.subset_size = args_dict["subset_size"]
    if args_dict["pre_type"] is not None:  # TODO implement?
        pre_type = args_dict["pre_type"]
        if pre_type == 'None':
            pre_type = None
        config.pre_type = pre_type

    if args_dict["split_fixed"] is not None:
        if args_dict["split_fixed"] == 'False':
            config.split_fixed = False
        elif args_dict["split_fixed"] == 'True':
            config.split_fixed = True
        else:
            raise ValueError('Incorrent value for argument "split_fixed":',
                             args_dict["split_fixed"],
                             'expected: "True" or "False"')
    else:
        config.split_fixed = False  # for backward compatibility with older configs
    # if args_dict["train_targets"] is not None:  # NOTE
    config.finetune.train_targets = args_dict["train_targets"]

    # prefix for names of resulting files (logs, checkpoints, CSVs)
    prefix = f'{config.finetune.epochs}_epochs'\
             f'_lr_{config.finetune.optimizer_params["lr"]}'
    #   f'_num_task_layers_{config.num_task_layers}'\
    if args_dict["pre_type"] is not None:
        prefix += f'_pre_type_{config.pre_type}'
    if hasattr(config.finetune, "loss2_weight"):
        prefix += f'_loss2_weight_{config.finetune.loss2_weight}'
    if hasattr(config.finetune, "subset_size"):
        prefix += f'_subset_size_{config.finetune.subset_size}'
    config.name = prefix + f'_{config.name}'

    milestone = int(config.finetune.epochs*(5/6))  # to keep it 50 for 60 epochs
    config.finetune.scheduler_params = {"milestones": [milestone], "gamma": 0.5}


def eval_only(config):
    torch.manual_seed(config.seed)

    # # pretrained_dir_path = 'datasets/PI_syn/processed'
    # pretrained_dir_path = 'datasets/PI_syn_subset_1K_for_pretrain/processed'
    # pretrained_mean, pretrained_std = torch.load(pretrained_dir_path + "/meta.pt")
    # print('pretrained_mean', pretrained_mean)
    # print('pretrained_std', pretrained_std)

    # checkpoints_path = 'checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_10_classes_improvements'
    # FIXME clean up
    checkpoints_path = 'checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_10_classes_dropout_0.9'
    for checkpoint_name in sorted(os.listdir(checkpoints_path)):
        if not checkpoint_name.endswith('.pth'):
            continue
        config.finetune.pretrained_weights =\
              os.path.join(checkpoints_path, checkpoint_name)
        print('checkpoint:', checkpoint_name)
            

    # config.finetune.pretrained_weights = \
    #         f"checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_5_classes/60_epochs_lr_0.001_subset_size_100000_round_133_PI_synfinetuning on PI_syn#Fri Nov 17 19:58:16 2023_simple_subset_100000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_5_classes/60_epochs_lr_0.001_subset_size_100000_round_133_PI_synfinetuning on PI_syn#Fri Nov 17 19:58:16 2023_simple_subset_100000_epoch_23_global_batch_35000_periodic.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_100K/60_epochs_lr_0.001_subset_size_100000_round_130_PI_synfinetuning on PI_syn#Thu Nov 16 23:38:26 2023_simple_subset_100000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_10_classes_balanced/60_epochs_lr_0.001_round_125_PI_synfinetuning on PI_syn#Thu Nov 16 18:30:15 2023_simple_subset_99000_epoch_44_global_batch_68000_periodic.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_10_classes_balanced/60_epochs_lr_0.001_round_125_PI_synfinetuning on PI_syn#Thu Nov 16 18:30:15 2023_simple_subset_99000_epoch_15_global_batch_23000_periodic.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_100K_classifier_fixed_10_classes/60_epochs_lr_0.001_subset_size_100000_round_115_PI_synfinetuning on PI_syn#Wed Nov 15 23:05:45 2023_simple_subset_100000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_classifier_fixed_10_classes_quantiles/60_epochs_lr_0.001_round_115_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Wed Nov 15 17:47:43 2023_simple_subset_9000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_classifier_fixed_10_classes/60_epochs_lr_0.001_round_105_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Fri Nov 10 11:02:25 2023_simple_subset_9000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_classifier_fixed/60_epochs_lr_0.001_round_98_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Fri Nov 10 08:40:19 2023_simple_subset_9000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_classifier/60_epochs_lr_0.001_round_93_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Tue Nov  7 23:08:34 2023_simple_subset_9000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_classifier/60_epochs_lr_0.001_round_93_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Tue Nov  7 23:08:34 2023_simple_subset_9000_epoch_14_global_batch_1900_periodic.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K_with_hard_label_noise/60_epochs_lr_0.001_round_92_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Tue Nov  7 18:54:38 2023_simple_subset_9000_last.pth"
    #         # f"checkpoints/finetune/copy_finetune_on_PI_syn_10K/60_epochs_lr_0.001_round_90_PI_syn_subset_10Kfinetuning on PI_syn_subset_10K#Tue Nov  7 15:02:05 2023_simple_subset_9000_epoch_22_global_batch_3000_best.pth"
    #         # f"checkpoints/finetune/copy_pretrained_on_100K_PI_syn_2_epochs/100_epochs_lr_0.0006_subset_size_100000_round_66_PI_syn_subset_100Kfinetuning on PI_syn_subset_100K#Sun Oct 29 22:11:24 2023_simple_subset_93600_epoch_2.pth"
    #         # f"checkpoints/pretrain/round_85_on_PI_syn_1K_for_pretrain/round_85pretrain_full_PI_syn_subset_1K_for_pretrain_Tg, K_Mon Nov  6 15:08:47 2023_epoch_100_global_batch_5000_periodic.pt"
    #         # f"checkpoints/pretrain/round_85_on_PI_syn_1K_for_pretrain/round_85pretrain_full_PI_syn_subset_1K_for_pretrain_Tg, K_Mon Nov  6 15:08:47 2023_epoch_10_global_batch_500_periodic.pt"
    #         # f"checkpoints/pretrain/round_85_on_PI_syn_1K_for_pretrain/round_85pretrain_full_PI_syn_subset_1K_for_pretrain_Tg, K_Mon Nov  6 15:08:47 2023_epoch_1_global_batch_50_periodic.pt"
    #         # f"checkpoints/pretrain/{config.pretrains[config.pre_type][1]}"
    #         # 'checkpoints/finetune/300_epochs_lr_0.01_pre_type_None_subset_size_750_round_55_PI_exp_new_07_03_2023finetuning on PI_exp_new_07_03_2023#Mon Oct 23 01:04:12 2023_simple_subset_750_last.pth'
        model = config.model
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)

        state_dict = load_pretrained_weights(model, config)
        # NOTE loading stats like this for now
        data_dict = torch.load(
                config.finetune.pretrained_weights,
                map_location=torch.device(config.device_index))
        # state_dict = data_dict['model_state_dict']  # FIXME remove
        pretrained_mean = data_dict['mean']
        pretrained_std = data_dict['std']
        print('pretrained_mean', pretrained_mean)
        print('pretrained_std', pretrained_std)

        # bins = data_dict['bins']  # NOTE
        
        model.load_state_dict(state_dict)

        target_name = config.finetune.target_name  # target chemical property

        if target_name is None:
                raise ValueError('set target (df column) name (e.g. "Tg")')

        test_dataset = config.finetune.eval_dataset
        print('test_dataset.mean', test_dataset.mean)
        print('test_dataset.std', test_dataset.std)

        # evaluate(test_dataset, save_preds=False, tb_logging=True, tb_writer=None,
        #          tb_iter_idx=None, log_tag='test', target_name=None,
        #          batch_size=8, model=None, dataset_mean=None, dataset_std=None,
        #          device=None, print_to_console=False)

        # eval_results = evaluate(test_dataset,
        eval_results = evaluate_classifier(test_dataset,  # FIXME change back
                                return_preds_df=True,
                                tb_logging=False,
                                log_tag='test',
                                target_name=target_name,
                                batch_size=config.batch_size,
                                model=model,
                                tgt_mean_to_denorm_preds=pretrained_mean,
                                tgt_std_to_denorm_preds=pretrained_std,
                                tgt_mean_to_norm_tgts=test_dataset.mean,
                                tgt_std_to_norm_tgts=test_dataset.std,
                                device=device,
                                print_to_console=True,
                                config=config,  # FIXME remove
                                # bins=bins,  # NOTE
                                )
        _, _, preds_df = eval_results

        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=8,
        #     shuffle=False,
        #     num_workers=4)
        
        # model.eval()
        # test_loss = 0.
        # test_MAE_denormalized = 0.
        # # val_progressbar = tqdm(test_loader, desc=f"Validation")
        # data_format = {"ID": [], f"{target_name}, pred": []}
        # return_table = pd.DataFrame(data_format)
        # # data_mean = dataset.mean.item()
        # # data_std = dataset.std.item()
        # # save_preds = True
        # with torch.no_grad():
        #     for i, test_data in enumerate(test_loader):
        #         test_data = test_data.to(device)
        #         pred = model(test_data).squeeze()
        #         # if save_preds:
        #         mol_ids = test_data.index.cpu().numpy().tolist()
        #         # pred_denormalized = (pred*data_std + data_mean).cpu().numpy().tolist()
        #         pred_denormalized_torch = (pred*pretrained_std + pretrained_mean)
        #         # print('mol_ids', mol_ids)
        #         # print('pred', pred)
        #         # print('pred_denormalized_torch', pred_denormalized_torch)
        #         # print('test_data.y', test_data.y)
        #         # assert False
        #         # print(f'pred: {float(pred_denormalized_torch.cpu().item()):.4f}, tgt: {float(test_data.y.item()):.4f}')
        #         pred_denormalized = pred_denormalized_torch.clone().cpu().numpy().tolist()
        #         data_batch = {
        #             "ID": mol_ids,
        #             f"{target_name}, pred": pred_denormalized}
        #         # print(data_batch)
        #         return_table = return_table.append(
        #             pd.DataFrame(data_batch), ignore_index=True)
                
        #         # if logging:
                
        #         # target_normalized = (test_data.y - test_dataset.mean)/test_dataset.std
        #         #     # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
        #         #     test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()
        #         # test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()
        #         test_MAE_denormalized += (pred_denormalized_torch-test_data.y).abs().sum().cpu().item()

        # # test_loss /= len(test_loader.dataset)
        # print('len(test_loader.dataset)', len(test_loader.dataset))
        # test_MAE_denormalized = test_MAE_denormalized / len(test_loader.dataset)
        # # test_MAE_denormalized = test_MAE*data_std
        # print('test_MAE_denormalized', test_MAE_denormalized)

        # eval_save_path = f'checkpoints/finetune/{config.name}_only_eval.csv'
        eval_save_path = f'checkpoints/finetune/{checkpoint_name}_only_eval.csv'
        preds_df.to_csv(eval_save_path, index=False)


def evaluate(test_dataset, return_preds_df=False, tb_logging=True,
             tb_writer=None, tb_iter_idx=None, log_tag='test',
             target_name=None, batch_size=8, model=None,
             tgt_mean_to_denorm_preds=None, tgt_std_to_denorm_preds=None,
             tgt_mean_to_norm_tgts=None, tgt_std_to_norm_tgts=None,
             device=None, print_to_console=False):
    """Performs evaluation (test or validation on "test_dataset").
    It can return predictions, save metrics to TensorBoard, or print
        them to console.

    Args:
        tgt_mean_to_denorm_preds: usually, it's the target mean the network
            was trained with.
        tgt_std_to_denorm_preds: usually, it's the target std the network
            was trained with.
        tgt_mean_to_norm_tgts: usually, it's the mean of the current dataset,
            or of the dataset it's a subset of.
        tgt_std_to_norm_tgts: usually, it's the std of the current dataset,
            or of the dataset it's a subset of.
        """
    if target_name is None and return_preds_df:
        raise ValueError('set target (df column) name (e.g. "Tg")')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    was_training = model.training
    model.eval()
    test_loss = 0.
    test_MAE_denormalized = 0.
    if torch.is_tensor(tgt_mean_to_denorm_preds):
        tgt_mean_to_denorm_preds = tgt_mean_to_denorm_preds.item()
    if torch.is_tensor(tgt_std_to_denorm_preds):
        tgt_std_to_denorm_preds = tgt_std_to_denorm_preds.item()

    # val_progressbar = tqdm(test_loader, desc=f"Validation")
    if return_preds_df:
        data_format = {"ID": [], f"{target_name}, pred": []}
        return_table = pd.DataFrame(data_format)
    else:
        return_table = None
    # counter = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_data = test_data.to(device)
            # temperature_mask = (test_data.y>=400)&((test_data.y<=600))  # NOTE
            pred = model(test_data).squeeze()
            # pred = pred[temperature_mask]  # NOTE
            # counter += len(pred)
            pred_denormalized_torch = (pred
                                       * tgt_std_to_denorm_preds
                                       + tgt_mean_to_denorm_preds)
            if return_preds_df:
                mol_ids = test_data.index.cpu().numpy().tolist()
                # mol_ids = test_data.index[temperature_mask].cpu().numpy().tolist()
                pred_denormalized = pred_denormalized_torch.clone().tolist()
                data_batch = {
                    "ID": mol_ids,
                    f"{target_name}, pred": pred_denormalized}
                return_table = return_table.append(
                    pd.DataFrame(data_batch), ignore_index=True)
            
            # batch_denorm_errors = (pred_denormalized_torch-test_data.y[temperature_mask]).abs()
            batch_denorm_errors = (pred_denormalized_torch-test_data.y).abs()
            test_MAE_denormalized += batch_denorm_errors.sum().cpu().item()

            target_normalized = ((test_data.y-tgt_mean_to_norm_tgts)
            # target_normalized = ((test_data.y[temperature_mask]-tgt_mean_to_norm_tgts)
                              / tgt_std_to_norm_tgts)
            test_loss += F.mse_loss(pred,
                                    target_normalized,
                                    reduction='sum').item()
    test_MAE_denormalized = test_MAE_denormalized / len(test_loader.dataset)
    # test_MAE_denormalized = test_MAE_denormalized / counter
    test_loss /= len(test_loader.dataset)
    # test_loss /= counter

    if print_to_console:
        print(f"{log_tag}: loss: {test_loss:.4f}"
              f", denorm. MAE: {test_MAE_denormalized:.4f}")
    
    if tb_logging:
        tb_writer.add_scalar(f'Loss_MSE/{log_tag}',
                             test_loss,
                             tb_iter_idx)
        # tb_writer.add_scalar(f'MAE/{log_tag}',
        #                      test_MAE,
        #                      tb_iter_idx)
        tb_writer.add_scalar(f'MAE/{log_tag}_denormalized',
                             test_MAE_denormalized,
                             tb_iter_idx)
    if was_training:
        model.train()
    return test_loss, test_MAE_denormalized, return_table

def evaluate_classifier(test_dataset, return_preds_df=False, tb_logging=True,
             tb_writer=None, tb_iter_idx=None, log_tag='test',
             target_name=None, batch_size=8, model=None,
             tgt_mean_to_denorm_preds=None, tgt_std_to_denorm_preds=None,
             tgt_mean_to_norm_tgts=None, tgt_std_to_norm_tgts=None,
             device=None, print_to_console=False, config=None,
            #  bins=None,
             ):
    """Performs evaluation (test or validation on "test_dataset").
    It can return predictions, save metrics to TensorBoard, or print
        them to console.

    Args:
        tgt_mean_to_denorm_preds: usually, it's the target mean the network
            was trained with.
        tgt_std_to_denorm_preds: usually, it's the target std the network
            was trained with.
        tgt_mean_to_norm_tgts: usually, it's the mean of the current dataset,
            or of the dataset it's a subset of.
        tgt_std_to_norm_tgts: usually, it's the std of the current dataset,
            or of the dataset it's a subset of.
        """
    if target_name is None and return_preds_df:
        raise ValueError('set target (df column) name (e.g. "Tg")')
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    was_training = model.training
    model.eval()
    test_loss = 0.
    test_MAE_denormalized = 0.
    if torch.is_tensor(tgt_mean_to_denorm_preds):
        tgt_mean_to_denorm_preds = tgt_mean_to_denorm_preds.item()
    if torch.is_tensor(tgt_std_to_denorm_preds):
        tgt_std_to_denorm_preds = tgt_std_to_denorm_preds.item()

    # val_progressbar = tqdm(test_loader, desc=f"Validation")
    if return_preds_df:
        data_format = {"ID": [], f"{target_name}, pred": []}
        return_table = pd.DataFrame(data_format)
    else:
        return_table = None

    # print(bins)
    # # NOTE
    # torch_bins = torch.tensor(np.concatenate([np.array([config.min_t]), bins]),
    #                           dtype=torch.float32, device=device)

    # counter = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_data = test_data.to(device)
            # temperature_mask = (test_data.y>=400)&((test_data.y<=600))  # NOTE
            pred = model(test_data).squeeze()

            # pred_denormalized = pred * tgt_std_to_denorm_preds + tgt_mean_to_denorm_preds
            pred_denormalized = torch.argmax(pred, dim=1)  # idx of bins
            # pred_idx = torch.argmax(pred, dim=1)  # idx of bins

            # NOTE
            # pred_denormalized = (torch_bins[pred_idx]
            #                      + (torch_bins[pred_idx+1]-torch_bins[pred_idx])/2)
            
            pred_denormalized = (pred_denormalized*config.bin_size
                                 + config.min_t
                                 + config.bin_size/2)

            pred_denormalized_torch = pred_denormalized.clone()
            # pred = pred[temperature_mask]  # NOTE
            # counter += len(pred)
            # pred_denormalized_torch = (pred
            #                            * tgt_std_to_denorm_preds
            #                            + tgt_mean_to_denorm_preds)
            if return_preds_df:
                mol_ids = test_data.index.cpu().numpy().tolist()
                # mol_ids = test_data.index[temperature_mask].cpu().numpy().tolist()
                pred_denormalized = pred_denormalized_torch.clone().tolist()
                data_batch = {
                    "ID": mol_ids,
                    f"{target_name}, pred": pred_denormalized}
                return_table = return_table.append(
                    pd.DataFrame(data_batch), ignore_index=True)
            
            # batch_denorm_errors = (pred_denormalized_torch-test_data.y[temperature_mask]).abs()
            batch_denorm_errors = (pred_denormalized_torch-test_data.y).abs()
            test_MAE_denormalized += batch_denorm_errors.sum().cpu().item()

            # target_normalized = ((test_data.y-tgt_mean_to_norm_tgts)
            # # target_normalized = ((test_data.y[temperature_mask]-tgt_mean_to_norm_tgts)
            #                   / tgt_std_to_norm_tgts)
            # test_loss += F.mse_loss(pred,
            #                         target_normalized,
            #                         reduction='sum').item()
    test_MAE_denormalized = test_MAE_denormalized / len(test_loader.dataset)
    # test_MAE_denormalized = test_MAE_denormalized / counter
    # test_loss /= len(test_loader.dataset)
    # test_loss /= counter

    if print_to_console:
        print(f"{log_tag}: loss: {test_loss:.4f}"
              f", denorm. MAE: {test_MAE_denormalized:.4f}")
    
    if tb_logging:
        # tb_writer.add_scalar(f'Loss_MSE/{log_tag}',
        #                      test_loss,
        #                      tb_iter_idx)
        # tb_writer.add_scalar(f'MAE/{log_tag}',
        #                      test_MAE,
        #                      tb_iter_idx)
        tb_writer.add_scalar(f'MAE/{log_tag}_denormalized',
                             test_MAE_denormalized,
                             tb_iter_idx)
    if was_training:
        model.train()
    return test_loss, test_MAE_denormalized, return_table


def finetune(config):
    torch.manual_seed(config.seed)

    dataset = config.finetune.dataset
    if config.finetune.eval:
        eval_dataset = config.finetune.eval_dataset
        eval_dataset.std = dataset.std
        eval_dataset.mean = dataset.mean
    else:
        eval_dataset = None

    model = config.model

    if not hasattr(config.finetune, 'pretrained_weights'):
        if config.pre_type is None:
            config.finetune.pretrained_weights = None
        else:
            config.finetune.pretrained_weights = \
            f"checkpoints/pretrain/{config.pretrains[config.pre_type][1]}"

    if config.finetune.pretrained_weights is not None:
        state_dict = load_pretrained_weights(model, config)
    else:
        state_dict = model.state_dict()
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    target_name = config.finetune.target_name  # target chemical property

    def train_on_split(train_dataset, val_dataset, test_dataset,
                       split_idx, log_path, scheduler, eval_dataset=None,
                       target_name=None):
        if target_name is None:
            raise ValueError('set target (df column) name (e.g. "Tg")')
        writer = SummaryWriter(log_path)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True)  # NOTE drop_last
        
        progressbar = tqdm(range(config.finetune.epochs),
                           desc=f"Training split {split_idx}")
        best_val_MAE = float('inf')
        # global_batch_count = 0
        for epoch in progressbar:
            t1 = time.time()
            train_loss = 0.0
            train_MAE = 0.0
            for batch_idx, data in enumerate(train_loader):
                # global_batch_count += 1
                data = data.to(device)
                target = (data.y - dataset.mean)/dataset.std
                # target = torch.zeros_like(data.y)
                # tgs = torch.split(data.tgs, data.num_tgs.tolist())
                # for mol_idx, mol_tgs in enumerate(tgs):
                #     target[mol_idx] = random.choice(mol_tgs)

                pred = model(data)
                
                # TODO change back to MSE
                # loss = F.l1_loss(pred.squeeze(), target, reduction='none')
                # loss = F.smooth_l1_loss(pred.squeeze(), target, beta=1.0, reduction='mean')
                loss = F.mse_loss(pred.squeeze(), target, reduction='none')  # FIXME divide by num batches if accumulating?
                train_loss += loss.detach().sum().item()
                # train_MAE += F.l1_loss(pred.squeeze(), target, reduction='sum').item()
                train_MAE += F.l1_loss(pred.squeeze(), target, reduction='sum').item()
                loss.mean().backward()
                # if global_batch_count % 4 == 0:  # 16 * 4 = 64
                optimizer.step()
                optimizer.zero_grad()
            # optimizer.step()  # NOTE: Gradient accumulation / full batch training
            # optimizer.zero_grad()
            scheduler.step()

            train_loss /= len(train_loader.dataset)  # FIXME drop_last is used!!! So this is incorrect
            assert dataset.std == train_loader.dataset.std
            train_MAE_denormalized = train_MAE / len(train_loader.dataset) * dataset.std.item()
            writer.add_scalar(f'Loss_MSE/train', train_loss, epoch)
            writer.add_scalar(f'MAE/train_denormalized', train_MAE_denormalized, epoch)
            t2 = time.time()

            # TODO uncomment
            # _, val_MAE, _ = evaluate(val_dataset, log_idx=epoch, writer=writer,
            #                          log_tag='val', target_name=target_name)

            # t3 = time.time()
            # # if val_MAE < best_val_MAE:
            # #     best_val_MAE = val_MAE
            # #     test_loss, test_MAE, return_table = evaluate(test_dataset, i, last=True)

            # # NOTE instead of combining splits at their best epochs,
            # # splits are combined all at the end of training
            # # And training duration must be adjusted to maximize average best results
            # _, test_MAE, _ = evaluate(test_dataset, log_idx=epoch, writer=writer, log_tag='test')
            # t4 = time.time()
            # # print(f'epoch time: train: {t2-t1:.2f}s, val {t3-t2:.2f}s, test {t4-t3:.2f}s')
            # loss_dict = {
            # #         "MSE": val_loss,
            # #         "MAE": val_MAE,
            #         "denorm. val MAE": f'{(val_MAE*dataset.std):.2f}',
            #         "denorm. test MAE": f'{(test_MAE*dataset.std):.2f}'
            #         }
            # progressbar.set_postfix(loss_dict)

        eval_results = evaluate(test_dataset,
                                return_preds_df=True,
                                tb_logging=False,
                                log_tag='test',
                                target_name=target_name,
                                batch_size=config.batch_size,
                                model=model,
                                tgt_mean_to_denorm_preds=dataset.mean,
                                tgt_std_to_denorm_preds=dataset.std,
                                tgt_mean_to_norm_tgts=dataset.mean,
                                tgt_std_to_norm_tgts=dataset.std,
                                device=device,
                                print_to_console=True)
        _, _, return_table = eval_results
        
        return_table_eval = None
        if eval_dataset is not None:
            eval_results = evaluate(eval_dataset,
                                    return_preds_df=True,
                                    tb_logging=False,
                                    log_tag='eval',
                                    target_name=target_name,
                                    batch_size=config.batch_size,
                                    model=model,
                                    tgt_mean_to_denorm_preds=dataset.mean,
                                    tgt_std_to_denorm_preds=dataset.std,
                                    tgt_mean_to_norm_tgts=dataset.mean,
                                    tgt_std_to_norm_tgts=dataset.std,
                                    device=device,
                                    print_to_console=True)
            _, _, return_table_eval = eval_results
        writer.close()
        return return_table, return_table_eval
            
    data_format = {"ID": [], f"{target_name}, pred": []}
    results_table = pd.DataFrame(data_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    eval_tables = []
    split_fn = k_fold_split_fixed if config.split_fixed else k_fold_split
    for split_idx, (train_dataset, val_dataset, test_dataset) \
        in enumerate(split_fn(dataset, config.finetune.n_splits)):
    # for split_idx, (train_dataset, val_dataset, test_dataset) \
    #     in enumerate(split_fn(dataset, config.finetune.n_splits)):
        # continue

        print(f"SPLIT {split_idx}")
        model.load_state_dict(state_dict, strict=False)
        # for p1, p2 in zip(state_dict.values(), orig_state_dict.values()):
        #     if (p1 != p2).sum() > 0:
        #         assert False
        print(f'split {split_idx} train_dataset size:', len(train_dataset))
        print(f'split {split_idx} val_dataset size:', len(val_dataset))
        print(f'split {split_idx} test_dataset size:', len(test_dataset))

        log_path = config.log_dir + "/finetune/" + config.name +\
            f'_split_{split_idx}'
        model.train()
        optimizer = torch.optim.RAdam(model.parameters(), **config.finetune.optimizer_params)
        # optimizer = RAdam(model.parameters(), **config.finetune.optimizer_params)
        # optimizer = torch.optim.SGD(model.parameters(), **config.finetune.optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.finetune.scheduler_params)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune.epochs, eta_min=1e-6)
        new_results_table, new_eval_table = train_on_split(
            train_dataset,
            val_dataset,
            test_dataset,
            split_idx,
            log_path,
            scheduler,
            eval_dataset,
            target_name)
        results_table = results_table.append(new_results_table, ignore_index = True)
        if config.finetune.eval:
            eval_tables.append(new_eval_table)
        # if split_idx == 0:
        cp_save_path = f'{config.checkpoints_dir}/finetune/'\
                        f'{config.name}_split_{split_idx}_last.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'mean': dataset.mean,
            'std': dataset.std,
            }, cp_save_path)

    results_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_final.csv'
    results_table.to_csv(results_save_path, index=False)
    if config.finetune.eval:
        eval_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_eval.pth'
        torch.save(eval_tables, eval_save_path)  # why saving dataframe as .pth??? list of dataframes for splits
    # writer.close()
    

