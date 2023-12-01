import time

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
    if args_dict["pre_type"] is not None:  # TODO implement?
        pre_type = args_dict["pre_type"]
        if pre_type == 'None':
            pre_type = None
        config.pre_type = pre_type

    # prefix for names of resulting files (logs, checkpoints, CSVs)
    prefix = f'{config.finetune.epochs}_epochs'\
             f'_lr_{config.finetune.optimizer_params["lr"]}'
    #   f'_num_task_layers_{config.num_task_layers}'\
    if args_dict["pre_type"] is not None:
        prefix += f'_pre_type_{config.pre_type}'
    if hasattr(config.finetune, "loss2_weight"):
        prefix += f'_loss2_weight_{config.finetune.loss2_weight}'
    config.name = prefix + f'_{config.name}'

    milestone = int(config.finetune.epochs*(5/6))  # to keep it 50 for 60 epochs
    config.finetune.scheduler_params = {"milestones": [milestone], "gamma": 0.5}


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

    def evaluate(test_dataset, log_idx=None, save_preds=False,
                 logging=True, writer=None, log_tag='test',
                 target_name=None):
        if target_name is None:
            raise ValueError('set target (df column) name (e.g. "Tg")')
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4)
        
        model.eval()
        test_loss = 0.
        test_MAE = 0.
        # val_progressbar = tqdm(test_loader, desc=f"Validation")
        data_format = {"ID": [], f"{target_name}, pred": []}
        return_table = pd.DataFrame(data_format)
        data_mean = dataset.mean.item()
        data_std = dataset.std.item()
        with torch.no_grad():
            for i, test_data in enumerate(test_loader):
                test_data = test_data.to(device)
                pred = model(test_data).squeeze()
                if save_preds:
                    mol_ids = test_data.index.cpu().numpy().tolist()
                    pred_denormalized = (pred*data_std + data_mean).cpu().numpy().tolist()
                    data_batch = {
                        "ID": mol_ids,
                        f"{target_name}, pred": pred_denormalized}
                    # print(data_batch)
                    return_table = return_table.append(
                        pd.DataFrame(data_batch), ignore_index=True)
                
                if logging:
                    target_normalized = (test_data.y - data_mean)/data_std
                    # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
                    test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()
                    test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()

        if logging:
            test_loss /= len(test_loader.dataset)
            test_MAE = test_MAE / len(test_loader.dataset)
            test_MAE_denormalized = test_MAE*data_std

            # print(f"{log_tag}: Loss: {test_loss:.4f}, MAE: {test_MAE:.4f}, denorm. MAE: {test_MAE_denormalized:.4f}")

            writer.add_scalar(f'Loss_MSE/{log_tag}', test_loss, log_idx)
            # writer.add_scalar(f'Loss_SmoothL1/{log_tag}', test_loss, log_idx)
            writer.add_scalar(f'MAE/{log_tag}', test_MAE, log_idx)
            writer.add_scalar(f'MAE/{log_tag}_denormalized', test_MAE_denormalized, log_idx)
        model.train()
        return test_loss, test_MAE, return_table

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
        test_loss, test_MAE, return_table = evaluate(
            test_dataset, save_preds=True, logging=False, target_name=target_name)
        return_table_eval = None
        if eval_dataset is not None:
            _, _, return_table_eval = evaluate(eval_dataset, split_idx,
                                               save_preds=True, logging=False,
                                               log_tag='eval', target_name=target_name)
        writer.close()
        return return_table, return_table_eval
            
    data_format = {"ID": [], f"{target_name}, pred": []}
    results_table = pd.DataFrame(data_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    eval_tables = []
    for split_idx, (train_dataset, val_dataset, test_dataset) \
        in enumerate(k_fold_split(dataset, config.finetune.n_splits)):
    # for split_idx, (train_dataset, val_dataset, test_dataset) \
    #     in enumerate(k_fold_split_fixed(dataset, config.finetune.n_splits)):
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
        # optimizer = torch.optim.RAdam(model.parameters(), **config.finetune.optimizer_params)
        optimizer = RAdam(model.parameters(), **config.finetune.optimizer_params)
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
        if split_idx == 0:
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
    

