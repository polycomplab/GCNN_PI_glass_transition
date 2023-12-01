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


def finetune_on_2_targets(config):
    torch.manual_seed(config.seed)

    dataset1 = config.finetune.dataset1
    dataset2 = config.finetune.dataset2
    # if config.finetune.eval:  # TODO support
    #     eval_dataset = config.finetune.eval_dataset
    #     eval_dataset.std = dataset.std
    #     eval_dataset.mean = dataset.mean
    # else:
    eval_dataset = None

    model = config.model
    if config.finetune.pretrained_weights is not None:
        state_dict = load_pretrained_weights(model, config)
    else:
        state_dict = model.state_dict()
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    target1_name = config.finetune.target1_name  # target chemical property
    target2_name = config.finetune.target2_name  # target chemical property

    def evaluate(test_dataset1, test_dataset2, log_idx=None, save_preds=False,
                 logging=True, writer=None, log_tag='test',
                 target1_name=None, target2_name=None):
        if target1_name is None:
            raise ValueError('set target1 (df column) name (e.g. "Tg")')
        if target2_name is None:
            raise ValueError('set target2 (df column) name (e.g. "Tg")')
        test_loader1 = DataLoader(
            test_dataset1,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=1)
        test_loader2 = DataLoader(
            test_dataset2,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=1)
        
        model.eval()
        test_loss1 = 0.
        test_loss2 = 0.
        test_MAE1 = 0.
        test_MAE2 = 0.
        # val_progressbar = tqdm(test_loader, desc=f"Validation")
        data1_format = {"ID": [], f"{target1_name}, pred": []}
        data2_format = {"ID": [], f"{target2_name}, pred": []}
        return_table1 = pd.DataFrame(data1_format)
        return_table2 = pd.DataFrame(data2_format)
        data1_mean = dataset1.mean.item()
        data2_mean = dataset2.mean.item()
        data1_std = dataset1.std.item()
        data2_std = dataset2.std.item()
        assert data1_std == test_loader1.dataset.std
        with torch.no_grad():
            for i, test_data in enumerate(test_loader1):
                test_data = test_data.to(device)
                target1_normalized = (test_data.y - data1_mean)/data1_std
                pred = model(test_data)[:, 0].squeeze()
                if save_preds:
                    mol_ids = test_data.index.cpu().numpy().tolist()
                    preds = (pred*data1_std+data1_mean).cpu().numpy().tolist()
                    data_batch = {
                        "ID": mol_ids,
                        f"{target1_name}, pred": preds}
                    # print(data_batch)
                    return_table1 = return_table1.append(
                        pd.DataFrame(data_batch), ignore_index=True)
                
                if logging:
                    # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
                    test_loss1 += F.mse_loss(pred, target1_normalized, reduction='sum').item()
                    test_MAE1 += F.l1_loss(pred, target1_normalized, reduction='sum').item()

            for i, test_data in enumerate(test_loader2):
                test_data = test_data.to(device)
                target2_normalized = (test_data.y - data2_mean)/data2_std
                pred = model(test_data)[:, 1].squeeze()
                if save_preds:
                    mol_ids = test_data.index.cpu().numpy().tolist()
                    preds = (pred*data2_std+data2_mean).cpu().numpy().tolist()
                    data_batch = {
                        "ID": mol_ids,
                        f"{target2_name}, pred": preds}
                    # print(data_batch)
                    return_table2 = return_table2.append(
                        pd.DataFrame(data_batch), ignore_index=True)
                
                if logging:
                    # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
                    test_loss2 += F.mse_loss(pred, target2_normalized, reduction='sum').item()
                    test_MAE2 += F.l1_loss(pred, target2_normalized, reduction='sum').item()

        if logging:
            test_loss1 /= len(test_loader1.dataset)
            test_loss2 /= len(test_loader2.dataset)
            test_loss_total = test_loss1 + test_loss2
            test_MAE1 = test_MAE1 / len(test_loader1.dataset)
            test_MAE2 = test_MAE2 / len(test_loader2.dataset)
            test_MAE1_denormalized = test_MAE1*data1_std
            test_MAE2_denormalized = test_MAE2*data2_std

            # print(f"{log_tag}: Loss: {test_loss:.4f}, MAE: {test_MAE:.4f}, denorm. MAE: {test_MAE_denormalized:.4f}")

            writer.add_scalar(f'Loss_MSE1/{log_tag}', test_loss1, log_idx)
            writer.add_scalar(f'Loss_MSE2/{log_tag}', test_loss2, log_idx)
            writer.add_scalar(f'Loss_MSE_total/{log_tag}', test_loss_total, log_idx)
            # writer.add_scalar(f'Loss_SmoothL1/{log_tag}', test_loss, log_idx)
            writer.add_scalar(f'MAE1/{log_tag}', test_MAE1, log_idx)
            writer.add_scalar(f'MAE2/{log_tag}', test_MAE2, log_idx)
            writer.add_scalar(f'MAE1/{log_tag}_denormalized', test_MAE1_denormalized, log_idx)
            writer.add_scalar(f'MAE2/{log_tag}_denormalized', test_MAE2_denormalized, log_idx)
        model.train()
        return test_loss1, test_MAE1, return_table1, return_table2  # NOTE not used anyway, maybe except for the tables

    def train_on_split(*, train_dataset1, val_dataset1, test_dataset1,
                       train_dataset2, val_dataset2, test_dataset2,
                       split_idx, log_path, optimizer, scheduler,
                    #    split_idx, log_path, optimizer1, optimizer2, scheduler1, scheduler2,
                       target1_name, target2_name, eval_dataset=None,
                       ):
        writer = SummaryWriter(log_path)
        train_loader1 = DataLoader(
            train_dataset1,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)  # NOTE drop_last
        train_loader2 = DataLoader(
            train_dataset2,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True)  # NOTE drop_last
        
        progressbar = tqdm(range(config.finetune.epochs),
                           desc=f"Training split {split_idx}")
        best_val_MAE = float('inf')
        for epoch in progressbar:
            t1 = time.time()
            train_one_epoch(train_loader1, train_loader2, model, device,
                    config, optimizer, scheduler, epoch, writer)
            # train_one_epoch_two_optims(train_loader1, train_loader2, model, device,
            #         config, optimizer1,  optimizer2, scheduler1, scheduler2, epoch, writer)
            # train_one_epoch_v2(train_loader1, train_loader2, model, device,
            #         config, optimizer, scheduler, epoch, writer)
            t2 = time.time()

            # _, val_MAE, _ = evaluate(val_dataset1, val_dataset2, log_idx=epoch, writer=writer,

            _ = evaluate(
                val_dataset1,
                val_dataset2,
                log_idx=epoch,
                writer=writer,
                log_tag='val',
                target1_name=target1_name,
                target2_name=target2_name)
            
            # t3 = time.time()
            # # if val_MAE < best_val_MAE:
            # #     best_val_MAE = val_MAE
            # #     test_loss, test_MAE, return_table = evaluate(test_dataset, i, last=True)

            # # NOTE instead of combining splits at their best epochs,
            # # splits are combined all at the end of training
            # # And training duration must be adjusted to maximize average best results
            # _, test_MAE, _ = evaluate(test_dataset, log_idx=epoch, writer=writer, log_tag='test')

        test_loss1, test_MAE1, return_table1, return_table2 = evaluate(
            test_dataset1,
            test_dataset2,
            save_preds=True,
            logging=False,
            target1_name=target1_name,
            target2_name=target2_name)
        
        return_table_eval = None
        # if eval_dataset is not None:  # TODO support
        #     _, _, return_table_eval = evaluate(eval_dataset, split_idx,
        #                                        save_preds=True, logging=False,
        #                                        log_tag='eval', target_name=target_name)
        writer.close()
        return return_table1, return_table2, return_table_eval
            
    data1_format = {"ID": [], f"{target1_name}, pred": []}
    data2_format = {"ID": [], f"{target2_name}, pred": []}
    results_table1 = pd.DataFrame(data1_format)
    results_table2 = pd.DataFrame(data2_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    eval_tables = []
    for split_idx, (split_dataset_1, split_dataset_2) \
        in enumerate(zip(k_fold_split_fixed(dataset1, config.finetune.n_splits),\
                         k_fold_split_fixed(dataset2, config.finetune.n_splits))):
        # in enumerate(zip(k_fold_split(dataset1, config.finetune.n_splits),\
                        #  k_fold_split(dataset2, config.finetune.n_splits))):
        train_dataset1, val_dataset1, test_dataset1 = split_dataset_1
        train_dataset2, val_dataset2, test_dataset2 = split_dataset_2
    # for split_idx, (train_dataset, val_dataset, test_dataset) \
    #     in enumerate(k_fold_split_fixed(dataset, config.finetune.n_splits)):
        # continue

        print(f"SPLIT {split_idx}")
        model.load_state_dict(state_dict, strict=False)  # TODO check if state_dict has changed
        # for p1, p2 in zip(state_dict.values(), orig_state_dict.values()):
        #     if (p1 != p2).sum() > 0:
        #         assert False
        print(f'split {split_idx} train_dataset1 size:', len(train_dataset1))
        print(f'split {split_idx} train_dataset2 size:', len(train_dataset2))
        print(f'split {split_idx} val_dataset1 size:', len(val_dataset1))
        print(f'split {split_idx} val_dataset2 size:', len(val_dataset2))
        print(f'split {split_idx} test_dataset1 size:', len(test_dataset1))
        print(f'split {split_idx} test_dataset2 size:', len(test_dataset2))

        log_path = config.log_dir + "/finetune/" + config.name +\
            f'_split_{split_idx}'
        model.train()
        # optimizer = torch.optim.RAdam(model.parameters(), **config.finetune.optimizer_params)
        optimizer = RAdam(model.parameters(), **config.finetune.optimizer_params)
        # optimizer1 = RAdam(model.parameters(), **config.finetune.optimizer_params)
        # optimizer2 = RAdam(model.parameters(), **config.finetune.optimizer_params)
        # optimizer = torch.optim.SGD(model.parameters(), **config.finetune.optimizer_params)  # FIXME
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.finetune.scheduler_params)
        # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, **config.finetune.scheduler_params)
        # scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer2, **config.finetune.scheduler_params)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune.epochs, eta_min=1e-6)
        new_results_table1, new_results_table2, new_eval_table = train_on_split(
            train_dataset1=train_dataset1,
            val_dataset1=val_dataset1,
            test_dataset1=test_dataset1,
            train_dataset2=train_dataset2,
            val_dataset2=val_dataset2,
            test_dataset2=test_dataset2,
            split_idx=split_idx,
            log_path=log_path,
            optimizer=optimizer,
            scheduler=scheduler,
            # optimizer1=optimizer1,
            # optimizer2=optimizer2,
            # scheduler1=scheduler1,
            # scheduler2=scheduler2,
            target1_name=target1_name,
            target2_name=target2_name,
            eval_dataset=eval_dataset,
            )
        results_table1 = results_table1.append(new_results_table1, ignore_index = True)
        results_table2 = results_table2.append(new_results_table2, ignore_index = True)
        # if config.finetune.eval:  # TODO support
        #     eval_tables.append(new_eval_table)

    cp_save_path1 = f'{config.checkpoints_dir}/finetune/{config.name}_{target1_name}_final.csv'
    cp_save_path2 = f'{config.checkpoints_dir}/finetune/{config.name}_{target2_name}_final.csv'
    results_table1.to_csv(cp_save_path1, index=False)
    results_table2.to_csv(cp_save_path2, index=False)
    # if config.finetune.eval:  # TODO support
    #     eval_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_eval.pth'
    #     torch.save(eval_tables, eval_save_path)  # why saving dataframe as .pth??? list of dataframes for splits
    # # writer.close()
    

def train_one_epoch(train_loader1, train_loader2, model, device,
                    config, optimizer, scheduler, epoch, writer):
    train_loss1 = 0.0
    train_loss2 = 0.0
    train_loss_total = 0.0
    train_MAE1 = 0.0
    train_MAE2 = 0.0
    actual_batch_size = 0  # takes gradient accumulation into account
    # num_batches_in_epoch = 0
    for data1, data2 in zip(train_loader1, train_loader2):
        data1 = data1.to(device)
        data2 = data2.to(device)
        target1 = (data1.y - train_loader1.dataset.mean)/train_loader1.dataset.std
        target2 = (data2.y - train_loader2.dataset.mean)/train_loader2.dataset.std
        # target1 = data1.y
        # num_batches_in_epoch += 1
        actual_batch_size += len(target1)  # NOTE same batch size for another target, due to zipping
        # target2 = data2.y
        # target = torch.zeros_like(data.y)
        # tgs = torch.split(data.tgs, data.num_tgs.tolist())
        # for mol_idx, mol_tgs in enumerate(tgs):
        #     target[mol_idx] = random.choice(mol_tgs)

        pred1 = model(data1)[:, 0]
        pred2 = model(data2)[:, 1]
        
        # TODO change back to MSE
        # loss = F.l1_loss(pred.squeeze(), target, reduction='none')
        # loss = F.smooth_l1_loss(pred.squeeze(), target, beta=1.0, reduction='mean')
        # loss1 = F.smooth_l1_loss(pred1.squeeze(), target1, reduction='none')
        loss1 = F.mse_loss(pred1.squeeze(), target1, reduction='none')  # FIXME divide by num batches if accumulating?
        # loss2 = F.smooth_l1_loss(pred2.squeeze(), target2, reduction='none')
        loss2 = F.mse_loss(pred2.squeeze(), target2, reduction='none')  # FIXME divide by num batches if accumulating?
        train_loss1 += loss1.detach().sum().item()
        train_loss2 += loss2.detach().sum().item()
        loss = config.finetune.loss1_weight*loss1 \
            + config.finetune.loss2_weight*loss2
        # loss = config.finetune.loss1_weight*loss1  # perm_only mode
        train_loss_total += loss.detach().sum().item()

        # train_MAE += F.l1_loss(pred.squeeze(), target, reduction='sum').item()
        train_MAE1 += F.l1_loss(pred1.squeeze(), target1, reduction='sum').item()
        train_MAE2 += F.l1_loss(pred2.squeeze(), target2, reduction='sum').item()
        
        loss.mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    # print('actual batch size:', actual_batch_size, 'num batches in epoch:', num_batches_in_epoch)
    # if (epoch+1) % 4 == 0:
    #     optimizer.step()  # NOTE: Gradient accumulation / full batch training
    #     optimizer.zero_grad()
    # optimizer.step()
    # optimizer.zero_grad()
    scheduler.step()

    train_loss1 /= actual_batch_size
    train_loss2 /= actual_batch_size
    train_loss_total /= actual_batch_size
    assert config.finetune.dataset2.std == train_loader2.dataset.std
    train_MAE1_denormalized = train_MAE1 / actual_batch_size *\
          config.finetune.dataset1.std.item()
    train_MAE2_denormalized = train_MAE2 / actual_batch_size *\
          config.finetune.dataset2.std.item()
    writer.add_scalar(f'Loss_MSE1/train', train_loss1, epoch)
    writer.add_scalar(f'Loss_MSE2/train', train_loss2, epoch)
    writer.add_scalar(f'Loss_MSE_total/train', train_loss_total, epoch)
    writer.add_scalar(f'MAE1/train_denormalized', train_MAE1_denormalized, epoch)
    writer.add_scalar(f'MAE2/train_denormalized', train_MAE2_denormalized, epoch)


def train_one_epoch_v2(train_loader1, train_loader2, model, device,
                    config, optimizer, scheduler, epoch, writer):
    train_loss1 = 0.0
    train_loss2 = 0.0
    train_MAE1 = 0.0
    train_MAE2 = 0.0
    actual_batch_size = 0  # takes gradient accumulation into account
    # num_batches_in_epoch = 0
    for data1, data2 in zip(train_loader1, train_loader2):

        # Permeability
        data1 = data1.to(device)
        # target1 = data1.y
        target1 = (data1.y - train_loader1.dataset.mean)/train_loader1.dataset.std
        # num_batches_in_epoch += 1
        actual_batch_size += len(target1)  # NOTE same batch size for another target, due to zipping
        pred1 = model(data1)[:, 0]
        loss1 = F.mse_loss(pred1.squeeze(), target1, reduction='none')  # FIXME divide by num batches if accumulating?
        train_loss1 += loss1.detach().sum().item()
        train_MAE1 += F.l1_loss(pred1.squeeze(), target1, reduction='sum').item()

        loss = config.finetune.loss1_weight*loss1
        loss.mean().backward()

        # Tg
        data2 = data2.to(device)
        # target2 = data2.y
        target2 = (data2.y - train_loader2.dataset.mean)/train_loader2.dataset.std
        pred2 = model(data2)[:, 1]
        loss2 = F.mse_loss(pred2.squeeze(), target2, reduction='none')  # FIXME divide by num batches if accumulating?
        train_loss2 += loss2.detach().sum().item()
        train_MAE2 += F.l1_loss(pred2.squeeze(), target2, reduction='sum').item()

        loss = config.finetune.loss2_weight*loss2
        loss.mean().backward()

    # print('actual batch size:', actual_batch_size, 'num batches in epoch:', num_batches_in_epoch)
    optimizer.step()  # NOTE: Gradient accumulation / full batch training
    optimizer.zero_grad()
    scheduler.step()

    train_loss1 /= actual_batch_size
    train_loss2 /= actual_batch_size
    train_loss_total = train_loss1 + train_loss2
    train_loss_total /= actual_batch_size
    train_MAE1_denormalized = train_MAE1 / actual_batch_size *\
          config.finetune.dataset1.std.item()
    train_MAE2_denormalized = train_MAE2 / actual_batch_size *\
          config.finetune.dataset2.std.item()
    writer.add_scalar(f'Loss_MSE1/train', train_loss1, epoch)
    writer.add_scalar(f'Loss_MSE2/train', train_loss2, epoch)
    writer.add_scalar(f'Loss_MSE_total/train', train_loss_total, epoch)
    writer.add_scalar(f'MAE1/train_denormalized', train_MAE1_denormalized, epoch)
    writer.add_scalar(f'MAE2/train_denormalized', train_MAE2_denormalized, epoch)



def train_one_epoch_two_optims(train_loader1, train_loader2, model, device,
                    config, optimizer1, optimizer2, scheduler1, scheduler2, epoch, writer):
    train_loss1 = 0.0
    train_loss2 = 0.0
    train_MAE1 = 0.0
    train_MAE2 = 0.0
    actual_batch_size = 0  # takes gradient accumulation into account
    num_batches_in_epoch = 0
    for data1, data2 in zip(train_loader1, train_loader2):

        # Permeability
        data1 = data1.to(device)
        # target1 = data1.y
        target1 = (data1.y - train_loader1.dataset.mean)/train_loader1.dataset.std
        num_batches_in_epoch += 1
        actual_batch_size += len(target1)  # NOTE same batch size for another target, due to zipping
        pred1 = model(data1)[:, 0]
        loss1 = F.mse_loss(pred1.squeeze(), target1, reduction='none')  # FIXME divide by num batches if accumulating?
        train_loss1 += loss1.detach().sum().item()
        train_MAE1 += F.l1_loss(pred1.squeeze(), target1, reduction='sum').item()

        loss = config.finetune.loss1_weight*loss1
        loss.mean().backward()
        optimizer1.step()  # NOTE: Gradient accumulation / full batch training
        optimizer1.zero_grad()
        scheduler1.step()

        # Tg
        data2 = data2.to(device)
        # target2 = data2.y
        target2 = (data2.y - train_loader2.dataset.mean)/train_loader2.dataset.std
        pred2 = model(data2)[:, 1]
        loss2 = F.mse_loss(pred2.squeeze(), target2, reduction='none')  # FIXME divide by num batches if accumulating?
        train_loss2 += loss2.detach().sum().item()
        train_MAE2 += F.l1_loss(pred2.squeeze(), target2, reduction='sum').item()

        loss = config.finetune.loss2_weight*loss2
        loss.mean().backward()
        optimizer2.step()  # NOTE: Gradient accumulation / full batch training
        optimizer2.zero_grad()
        scheduler2.step()

    # print('actual batch size:', actual_batch_size, 'num batches in epoch:', num_batches_in_epoch)
    # optimizer.step()  # NOTE: Gradient accumulation / full batch training
    # optimizer.zero_grad()
    # scheduler.step()

    train_loss1 /= actual_batch_size
    train_loss2 /= actual_batch_size
    train_loss_total = train_loss1 + train_loss2
    train_loss_total /= actual_batch_size
    train_MAE1_denormalized = train_MAE1 / actual_batch_size *\
          config.finetune.dataset1.std.item()
    train_MAE2_denormalized = train_MAE2 / actual_batch_size *\
          config.finetune.dataset2.std.item()
    writer.add_scalar(f'Loss_MSE1/train', train_loss1, epoch)
    writer.add_scalar(f'Loss_MSE2/train', train_loss2, epoch)
    writer.add_scalar(f'Loss_MSE_total/train', train_loss_total, epoch)
    writer.add_scalar(f'MAE1/train_denormalized', train_MAE1_denormalized, epoch)
    writer.add_scalar(f'MAE2/train_denormalized', train_MAE2_denormalized, epoch)
