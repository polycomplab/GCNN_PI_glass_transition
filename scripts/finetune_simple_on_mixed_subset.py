import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from radam import RAdam
from data.KGNNDataLoader import DataLoader
from data.data_splitting import split_train_subset, split_train_val
from utils import load_pretrained_weights


def finetune_simple_on_mixed_subset(config):
    torch.manual_seed(config.seed)

    # dataset = config.finetune.dataset
    # if config.finetune.eval:
    #     eval_dataset = config.finetune.eval_dataset
    #     eval_dataset.std = dataset.std
    #     eval_dataset.mean = dataset.mean
    # else:
    #     eval_dataset = None

    # dataset_mean = dataset.mean
    # dataset_std = dataset.std
    # print(f'current dataset mean {dataset_mean:.2f} and std {dataset_std:.2f}')

    model = config.model

    if not hasattr(config.finetune, 'pretrained_weights'):
        if config.pre_type is None:
            config.finetune.pretrained_weights = None
        else:
            config.finetune.pretrained_weights = \
            f"checkpoints/finetune/{config.pretrains[config.pre_type][1]}"
            # f"checkpoints/pretrain/{config.pretrains[config.pre_type][1]}"

    if config.finetune.pretrained_weights is not None:
        state_dict = load_pretrained_weights(model, config)

        # TODO maybe use eventually:
        # loaded_state_dict = torch.load(
        #     config.finetune.pretrained_weights,
        #     map_location=torch.device(config.device_index))
        # dataset_mean = loaded_state_dict['mean']
        # dataset_std = loaded_state_dict['std']
        # print(f'loaded mean {dataset_mean:.2f} and std {dataset_std:.2f}')
        # del loaded_state_dict
    else:
        state_dict = model.state_dict()
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    target_name = config.finetune.target_name  # target chemical property
            
    data_format = {"ID": [], f"{target_name}, pred": []}
    results_table = pd.DataFrame(data_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    # eval_tables = []
    # train_dataset, test_dataset = \
    #     split_train_val(dataset, test_size=config.finetune.test_subset_size)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=700)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=750)

    # split_fn = k_fold_split_fixed if config.split_fixed else k_fold_split
    # for split_idx, (train_dataset, val_dataset, test_dataset) \
    #     in enumerate(split_fn(dataset, config.finetune.n_splits)):
    # for split_idx, (train_dataset, val_dataset, test_dataset) \
    #     in enumerate(split_fn(dataset, config.finetune.n_splits)):
        # continue

    # print(f"SPLIT {split_idx}")
    model.load_state_dict(state_dict, strict=False)
    # for p1, p2 in zip(state_dict.values(), orig_state_dict.values()):
    #     if (p1 != p2).sum() > 0:
    #         assert False

    # for mixed training dataset only
    train_dataset = config.finetune.train_dataset
    test_dataset = config.finetune.test_dataset

    # target dataset mean & std
    dataset_mean = train_dataset.ds1.mean
    dataset_std = train_dataset.ds1.std

    print(f'train_dataset size:', len(train_dataset))
    # print(f'split {split_idx} val_dataset size:', len(val_dataset))
    print(f'test_dataset size:', len(test_dataset))

    log_path = config.log_dir + "/finetune/" + config.name +\
        f'_simple_subset_{len(train_dataset)}'
    model.train()
    optimizer = torch.optim.RAdam(model.parameters(), **config.finetune.optimizer_params)
    # optimizer = RAdam(model.parameters(), **config.finetune.optimizer_params)
    # optimizer = torch.optim.SGD(model.parameters(), **config.finetune.optimizer_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, **config.finetune.scheduler_params)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=config.finetune.optimizer_params['lr'],
    #     total_steps=config.epochs,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune.epochs, eta_min=1e-6)
    # new_results_table, new_eval_table = train_on_split(
    new_results_table = train_on_subset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        log_path=log_path,
        scheduler=scheduler,
        config=config,
        device=device,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        model=model,
        optimizer=optimizer,
        target_name=target_name)
    results_table = results_table.append(new_results_table, ignore_index = True)
    # if config.finetune.eval:
    #     eval_tables.append(new_eval_table)
    # if split_idx == 0:
    cp_save_path = (f'{config.checkpoints_dir}/finetune/'
                    f'{config.name}_simple_subset_'
                    f'{len(train_dataset)}_last.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset_mean,
        # 'mean': dataset.mean,
        'std': dataset_std,
        # 'std': dataset.std,
        }, cp_save_path)

    results_save_path = (f'{config.checkpoints_dir}/finetune/'
                         f'{config.name}_final.csv')
    results_table.to_csv(results_save_path, index=False)
    # if config.finetune.eval:
    #     eval_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_eval.pth'
    #     torch.save(eval_tables, eval_save_path)  # why saving dataframe as .pth??? list of dataframes for splits
    # writer.close()
    
# def train_on_split(train_dataset, val_dataset, test_dataset,
def train_on_subset(*, train_dataset, test_dataset, log_path, scheduler,
                    config, device, dataset_mean, dataset_std,
                    model, optimizer, target_name):
    writer = SummaryWriter(log_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)  # NOTE drop_last
    
    progressbar = tqdm(range(1, config.finetune.epochs+1))
    # progressbar = tqdm(range(1, config.finetune.epochs+1),
    #                     desc=f"Subset size {config.finetune.subset_size}")
    best_val_MAE = float('inf')
    # global_batch_count = 0
    for epoch in progressbar:
        t1 = time.time()
        model.train()
        # train_loss = 0.0
        main_train_loss = 0.0
        extra_train_loss = 0.0
        # train_MAE = 0.0
        main_train_MAE = 0.0
        extra_train_MAE = 0.0
        main_mols_count = 0
        extra_mols_count = 0
        for batch_idx, data in enumerate(train_loader):
            # global_batch_count += 1
            data = data.to(device)
            target = (data.y - dataset_mean)/dataset_std
            # target = torch.zeros_like(data.y)
            # tgs = torch.split(data.tgs, data.num_tgs.tolist())
            # for mol_idx, mol_tgs in enumerate(tgs):
            #     target[mol_idx] = random.choice(mol_tgs)

            pred = model(data)

            main_mols_count += data.is_exp_last.sum()
            extra_mols_count += (len(data)-data.is_exp_last.sum())
            
            main_loss = F.mse_loss(
                pred.squeeze()[data.is_exp_last],
                target[data.is_exp_last],
                reduction='none')
            
            extra_loss = F.mse_loss(
                pred.squeeze()[~data.is_exp_last],
                target[~data.is_exp_last],
                reduction='none')

            # TODO change back to MSE
            # loss = F.l1_loss(pred.squeeze(), target, reduction='none')
            # loss = F.smooth_l1_loss(pred.squeeze(), target, beta=1.0, reduction='mean')
            # loss = F.mse_loss(pred.squeeze(), target, reduction='none')  # FIXME divide by num batches if accumulating?
            main_train_loss += main_loss.detach().sum().item()
            extra_train_loss += extra_loss.detach().sum().item()
            # train_loss += loss.detach().sum().item()
            # train_MAE += F.l1_loss(pred.squeeze(), target, reduction='sum').item()
            main_train_MAE += F.l1_loss(
                pred.squeeze()[data.is_exp_last],
                target[data.is_exp_last],
                reduction='sum').item()
            extra_train_MAE += F.l1_loss(
                pred.squeeze()[~data.is_exp_last],
                target[~data.is_exp_last],
                reduction='sum').item()
            
            # extra_loss_weight = 0.1  # NOTE tune here
            extra_loss_weight = float(config.finetune.loss2_weight)
            loss = main_loss.mean() + extra_loss_weight*extra_loss.mean()
            loss.backward()
            # loss.mean().backward()
            # if global_batch_count % 4 == 0:  # 16 * 4 = 64
            optimizer.step()
            optimizer.zero_grad()
        # optimizer.step()  # NOTE: Gradient accumulation / full batch training
        # optimizer.zero_grad()
        scheduler.step()

        # train_loss /= (len(train_loader) * config.batch_size)
        main_train_loss /= main_mols_count
        extra_train_loss /= extra_mols_count
        # assert dataset_std == train_loader.dataset.std
        # train_MAE_denormalized = (train_MAE
        #                           / (len(train_loader) * config.batch_size)
        #                           * dataset_std.item())
        main_train_MAE_denormalized = (main_train_MAE
                                       / main_mols_count
                                       * dataset_std.item())
        extra_train_MAE_denormalized = (extra_train_MAE
                                        / extra_mols_count
                                        * dataset_std.item())
        # writer.add_scalar(f'Loss_MSE/train', train_loss, epoch)
        # writer.add_scalar(f'MAE/train_denormalized', train_MAE_denormalized, epoch)
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
        # test_loss, test_MAE, return_table = evaluate(
        # test_loss, test_MAE, return_table = \
    test_loss, test_MAE = \
        evaluate(
            test_dataset=test_dataset,
            save_preds=False,
            target_name=target_name,
            config=config,
            model=model,
            dataset_mean=dataset_mean,
            dataset_std=dataset_std,
            device=device)
    print(f'epoch {epoch:2d}'
        #   f', train loss: {train_loss:.4f}'
            f', main train loss: {main_train_loss:.4f}'
            f', extra train loss: {extra_train_loss:.4f}'
            f', test loss: {test_loss:.4f}'
        #   f', train MAE: {train_MAE_denormalized:.2f}K'
            f', main train MAE: {main_train_MAE_denormalized:.2f}K'
            f', extra train MAE: {extra_train_MAE_denormalized:.2f}K'
            
            f', test MAE: {test_MAE*dataset_std.item():.2f}K'
            f', LR: {optimizer.param_groups[0]["lr"]:.6f}')
    # return_table_eval = None
    # if eval_dataset is not None:
    #     _, _, return_table_eval = evaluate(eval_dataset, split_idx,
    #                                        save_preds=True, logging=False,
    #                                        log_tag='eval', target_name=target_name)
    writer.close()
    # return return_table#, return_table_eval
    

def evaluate(*, test_dataset, save_preds, target_name, config, model,
             dataset_mean, dataset_std, device):
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4)
    
    model.eval()
    test_loss = 0.
    test_MAE = 0.
    # val_progressbar = tqdm(test_loader, desc=f"Validation")
    # data_format = {"ID": [], f"{target_name}, pred": []}
    # return_table = pd.DataFrame(data_format)
    data_mean = dataset_mean.item()
    data_std = dataset_std.item()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_data = test_data.to(device)
            pred = model(test_data).squeeze()
            # if save_preds:
            #     mol_ids = test_data.index.cpu().numpy().tolist()
            #     pred_denormalized = (pred*data_std + data_mean).cpu().numpy().tolist()
                # data_batch = {
                #     "ID": mol_ids,
                #     f"{target_name}, pred": pred_denormalized}
                # # print(data_batch)
                # return_table = return_table.append(
                #     pd.DataFrame(data_batch), ignore_index=True)
            
            target_normalized = (test_data.y - data_mean)/data_std
            # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
            test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()
            test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    test_MAE /= len(test_loader.dataset)
    test_MAE_denormalized = test_MAE*data_std
    # print(f"Test loss: {test_loss:.4f}, test MAE (Tg, K): {test_MAE_denormalized:.4f}")
    return test_loss, test_MAE#, return_table

