import time
import datetime
import argparse
import importlib.util
import random
import sys
import math
import os
from collections import defaultdict
import statistics

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch_geometric as pyg
from radam import RAdam
import pandas as pd
import numpy as np

from KGNNDataLoader import DataLoader
from ModifiedGatedGraphConv import ModifiedGatedGraphConv
from CustomConv import CustomConv
from pretrain import pretrain
from inference import run_inference

#from Model import Model
sys.path.append("./HGP-SL")
from models import Model

def train(config, args):

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    config.dataset = config.dataset.shuffle()

    # const_for_normalized_metric = \
    #     (config.dataset.data.y - config.dataset.data.y.mean()).std().item()
    # print(f'{const_for_normalized_metric:10.8f}')
    NUM_SPLITS = 10

    for split_idx in range(NUM_SPLITS):

        # this splitting scheme doesn't use all available data for test 
        # but makes splits to be of equal size hence comparable
        # idx0 = split_idx * config.test_size
        # idx1 = idx0 + config.test_size

        # this splitting scheme uses all available data for test
        # but the last test split is smaller (and last train split is bigger) than all previous splits
        max_test_size = math.ceil(len(config.dataset) / NUM_SPLITS)
        idx0 = split_idx * max_test_size
        idx1 = min(idx0 + max_test_size, len(config.dataset))

        test_dataset = config.dataset[idx0: idx1]
        if idx0 > 0:
            train_before = slice(None, idx0).indices(len(config.dataset))
            train_after = slice(idx1, None).indices(len(config.dataset))
            train_idx = torch.tensor(
                list(range(*train_before)) + list(range(*train_after))
            )
            train_dataset = config.dataset[train_idx]
        else:
            train_dataset = config.dataset[idx1:]

        # That's what it was without splits. It's the same as if split_idx == 0
        # test_dataset = config.dataset[:config.test_size]
        # train_dataset = config.dataset[config.test_size:]

        print(f'dataset size: train {len(train_dataset)}, test {len(test_dataset)}, '
              f'total {len(train_dataset) + len(test_dataset)}')

        target_index = config.target_index
        num_targets = 1

        std, mean = torch.std_mean(train_dataset.data.y)
        print('mean', mean.item(), 'std', std.item())

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

        runs_dir = config.tb_logs_dir
        conv_name = f'_{config.conv_type}_{config.num_convs}Convs_{config.layers_in_conv}Layers'
        if config.conv_type == pyg.nn.DNAConv.__name__:
            conv_name += f'_{config.num_heads}Heads'
            conv_name += f'_{config.num_groups}Groups'
        date_time = datetime.datetime.now().strftime('_%d-%b-%Y_%X_%f')
        run_name = config_filename + conv_name + date_time
        if config.add_to_name is not None:  # TODO
            run_name += f'_{config.add_to_name}'

        # run_name += '_train_after_pretrained_on_100k'  # FIXME
        # run_name += '_without_pretraining'

        run_name += '_N_split_idx_' + str(split_idx)

        if args.logs_suffix is not None:
            run_name += '_' + args.logs_suffix  # TODO

        print(f'run_name is {run_name}')

        if config.disable_logs:
            writer = None
            print('tensorboard logs disabled')
        else:
            full_run_name = runs_dir + run_name
            writer = SummaryWriter(full_run_name)

        if not torch.cuda.is_available():
            raise RuntimeError('no gpu')
        device = torch.device(f'cuda:{config.device_index}')

        hop_encoding_size = config.max_hops

        # torch.manual_seed(random.randint(100, 10000))

        model = Model(config.conv_type,
                      config.num_convs,
                      config.layers_in_conv,
                      config.num_channels,
                      config.use_embedding,
                      # test_dataset[0].num_node_features,  # be careful here when finetuning
                      train_dataset[0].num_node_features,  # be careful here when finetuning
                      config.num_heads,
                      config.use_nodetype_coeffs,
                      config.num_node_types,
                      config.num_edge_types,
                      config.use_jumping_knowledge,
                      config.embedding_size,
                      config.use_edgetype_coeffs,
                      config.use_bias_for_update,
                      config.global_aggrs,
                      config.use_dropout,
                      config.num_groups,
                      config.num_fc_layers,
                      config.neighbors_aggr,
                      config.dropout_p,
                      hop_encoding_size,
                      num_targets
                      ).to(device)

        if args.pretrained_weights is not None:
            model.load_state_dict(torch.load(args.pretrained_weights))
            print('Pretrained weights were loaded')

        optimizer = RAdam(model.parameters(),  # better than Adam, more stable learning
                          lr=config.lr,
                          weight_decay=config.weight_decay)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.5)

        def evaluate():
            out_d = None
            out_ids = None

            model.eval()
            test_loss = 0.
            test_MAE = 0.
            test_relative_error = 0.
            test_MSE_denormalized = 0.
            std_gpu = std.to(device)  # TODO
            mean_gpu = mean.to(device)
            with torch.no_grad():
                for i, test_data in enumerate(test_loader):
                    test_data_normalized = test_data.clone()
                    test_data_normalized.y = (test_data_normalized.y - mean) / std
                    test_data = test_data.to(device)
                    test_data_normalized = test_data_normalized.to(device)
                    out = model(test_data_normalized)
                    out = out[:, target_index].unsqueeze(1)


                    out_denormalized = out * std_gpu + mean_gpu
                    out_denormalized = out_denormalized.squeeze()
                    out_d_numpy = out_denormalized.cpu().numpy()

                    if out_d is not None: # never runs if testdata size < batch size
                        out_d = np.concatenate([out_d, out_d_numpy])
                    else:
                        out_d = out_d_numpy

                    out_ids_numpy = test_data.target_id.cpu().numpy()
                    if out_ids is not None:
                        out_ids = np.concatenate([out_ids, out_ids_numpy])
                    else:
                        out_ids = out_ids_numpy


                    test_data.y = test_data.y.unsqueeze(1)
                    test_target = test_data_normalized.y.unsqueeze(1)

                    # print('target ids', test_data.target_id)

                    test_loss += F.mse_loss(out,
                                            test_target,
                                            reduction='sum').item()
                    test_MAE += F.l1_loss(out,
                                          test_target,
                                          reduction='sum').item()
                    test_relative_error += (torch.abs(out * std_gpu + mean_gpu - test_data.y) / test_data.y).sum().item()
                    test_MSE_denormalized += ((out * std_gpu + mean_gpu - test_data.y) ** 2).sum().item()

            test_loss /= len(test_loader.dataset)
            test_MAE /= len(test_loader.dataset)
            test_relative_error /= len(test_loader.dataset)
            test_MSE_denormalized /= len(test_loader.dataset)
            return test_loss, test_MAE, test_relative_error, test_MSE_denormalized, out_d, out_ids

        best_test_MAE = float('inf')

        # train loop
        for epoch in range(1, config.epochs + 1):
            print(f'epoch {epoch:03d}')
            model.train()
            train_loss = 0.
            train_MAE = 0.
            prev_iters = (epoch - 1) * len(train_loader)
            # print(f'Split index {split_idx}')
            for i, data in enumerate(train_loader):

                data.y = (data.y - mean) / std
                data = data.to(device)
                real_batchsize = data.y.shape[0]

                global_iteration = prev_iters + i

                optimizer.zero_grad()
                out = model(data)

                target = data.y.unsqueeze(1)

                out = out[:, target_index].unsqueeze(1)

                loss = F.mse_loss(out, target, reduction='none')
                batch_summed_loss = loss.sum().item()
                batch_mean_loss = batch_summed_loss / real_batchsize
                train_loss += batch_summed_loss

                batch_MAE = F.l1_loss(out, target, reduction='none')
                batch_summed_MAE = batch_MAE.sum().item()
                batch_mean_MAE = batch_summed_MAE / real_batchsize
                denomalized_batch_MAE = batch_mean_MAE * std
                train_MAE += batch_mean_MAE
                print(f'train | split {split_idx}, epoch {epoch}, batch {i}, '
                # print(f'train | epoch {epoch}, batch {i}, '
                      f'batch loss (MSE) {batch_mean_loss:7.4f}, '
                      f'batch MAE {batch_mean_MAE:7.4f}, '
                      f'denormalized batch MAE {denomalized_batch_MAE:7.4f}')
                if not config.disable_logs:
                    writer.add_scalar('Batch_Loss_MSE/train', batch_mean_loss, global_iteration)
                    writer.add_scalar('Batch_MAE/train', denomalized_batch_MAE, global_iteration)
                loss.mean().backward()
                optimizer.step()

            if not config.disable_logs:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar('LR', current_lr, epoch)

            # test_loss, test_MAE, test_relative_error, test_MSE_denormalized = evaluate()
            test_loss, test_MAE, test_relative_error, test_MSE_denormalized, out_d, out_ids = evaluate()

            test_MAE_denormalized = test_MAE * std
            test_sqrt_MSE_denormalized = math.sqrt(test_MSE_denormalized)
            print(f'test | epoch {epoch}, '
                  f'loss (MSE)  {test_loss:7.4f}, '
                  f'test MAE  {test_MAE:7.4f}, '
                  f'test MAE denormalized {test_MAE_denormalized:7.4f}, '
                  f'test_relative_error {test_relative_error:7.4f}, '
                  f'test_sqrt_mse_denormalized {test_sqrt_MSE_denormalized:7.4f}')
            if not config.disable_logs:
                writer.add_scalar('Loss_MSE/test', test_loss, epoch)
                writer.add_scalar('MAE/test', test_MAE, epoch)
                writer.add_scalar('MAE/test_denormalized', test_MAE_denormalized, epoch)
                writer.add_scalar('MAE/test_relative_error', test_relative_error, epoch)
                writer.add_scalar('MAE/test_MSE_denormalized', test_MSE_denormalized, epoch)
                writer.add_scalar('MAE/test_sqrt_MSE_denormalized', test_sqrt_MSE_denormalized, epoch)

                scheduler.step()
                # scheduler.step(test_MAE)

            if test_MAE < best_test_MAE:
                best_test_MAE = test_MAE
                cp_save_path = f'{config.checkpoints_dir}{run_name}_best.pt'
                torch.save(model.state_dict(), cp_save_path)
                print(f'saved the model as {run_name}_best.pt')

            if epoch == config.epochs:
                cp_save_path = f'{config.checkpoints_dir}{run_name}_at_last_epoch.pt'
                torch.save(model.state_dict(), cp_save_path)
                print(f'saved the model as {run_name}_at_last_epoch.pt')

                df_out = pd.DataFrame({'pred': out_d, 'id': out_ids})
                # csv_name = f'pretrained_on_Askadsky100k_trained_on_realPI'  # TODO
                # csv_name += f'split_{split_idx}'
                csv_name = f'split_{split_idx}'
                subdir_name = 'pre_on_Askadsky_then_trained_and_tested_on_Base_20_03'
                # subdir_name = 'pre_on_Askadsky_not_finetuned_tested_on_Base_20_03'
                save_dir_path = f'/storage/Batyr/graphNets/test_csvs/{subdir_name}/{args.logs_suffix}'
                os.makedirs(save_dir_path, exist_ok=True)
                df_out.to_csv(f'{save_dir_path}/{csv_name}.csv')

        if not config.disable_logs:
            writer.close()

        print(f'best MAE on test: {best_test_MAE:7.4f}, denormalized: {best_test_MAE * std:7.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument('--pretraining_dataset', type=str,
                        help='Dataset for pretraining',
                        choices=['QM9', 'Askadsky', 'new_data'])
    parser.add_argument('--pretrained_weights', type=str,
                        help='Absolute path to checkpoint file; for finetuning')
    parser.add_argument('--weights_for_testing', type=str,
                        help='Absolute path to checkpoint file; for running inference')
    parser.add_argument('--logs_suffix', type=str,
                        help='For discriminating runs in Tensorboard logs')
    args = parser.parse_args()
    config_name = args.config

    modes = [args.train, args.pretrain, args.test]
    if sum(modes) != 1:
        raise ValueError('Specify \'train\' or \'pretrain\' or \'test\' flag')

    if args.pretrain and args.pretraining_dataset is None:
        raise ValueError('Specify the dataset for pretraining')
    if args.train and args.pretraining_dataset is not None:
        raise ValueError('The dataset is for pretraining only')

    if args.pretrained_weights is not None and not args.train:
        raise ValueError('Pretrained weights flag is for train mode only')

    if args.pretrained_weights is not None:
        if not os.path.isfile(args.pretrained_weights):
            raise ValueError('Pretrained weights checkpoint file not found')

    if args.weights_for_testing is not None and not args.test:
        raise ValueError('weights_for_testing flag is for test mode only')

    if args.weights_for_testing is not None:
        if not os.path.isfile(args.weights_for_testing):
            raise ValueError('Weights checkpoint file not found')

    config_filename = config_name.split('/')[-1]
    spec = importlib.util.spec_from_file_location("config", config_name)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if config_filename.endswith('.py'):
        config_filename = config_filename[:-3]
    config.config_filename = config_filename

    allowed_convs = [pyg.nn.GatedGraphConv.__name__,
                     ModifiedGatedGraphConv.__name__,
                     "OneGNNConv",
                     CustomConv.__name__,
                     pyg.nn.DNAConv.__name__,
                     pyg.nn.GCNConv.__name__]
    if config.conv_type not in allowed_convs:
        raise ValueError(f'conv_type not in {allowed_convs}')

    aggregation_ops = ['sum', 'mean', 'max', 'att']
    for aggr in config.global_aggrs:
        if aggr not in aggregation_ops:
            raise ValueError(f'global_aggr {aggr} not in {aggregation_ops}')

    if args.train:
        train(config, args)
    elif args.pretrain:
        pretrain(config, args)
    elif args.test:
        run_inference(config, args)
