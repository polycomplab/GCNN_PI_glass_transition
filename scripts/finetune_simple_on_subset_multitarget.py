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


def finetune_simple_on_subset_multitarget(config):
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
    if len(train_targets_split) == 1:
        cp_load_path = None
    else:
        cp_load_prefix = os.path.join(config.checkpoints_dir,
                                      f'{train_targets_split[1]}')
        cp_load_path = f'{cp_load_prefix}_last.pth'
        pretrain_targets = train_targets_split[1].split('_and_')

    train_targets = train_targets_split[0].split('_and_')


    save_preds = config.finetune.save_preds
    # if save_preds:
    #     data_format = {"ID": [], f"{target_name}, pred": []}
    #     results_table = pd.DataFrame(data_format)
    # eval_format = {"ID": [], "Tg, pred": []}
    # eval_tables = []
    train_dataset, test_dataset = \
        split_train_val(dataset, test_size=config.finetune.test_subset_size)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=700)
        # split_train_subset(dataset, train_size=config.finetune.subset_size, max_train_size=750)
    if (hasattr(config.finetune, 'subset_size')
        and config.finetune.subset_size is not None):
        train_dataset = split_subindex(train_dataset, config.finetune.subset_size)
    
    print(f'train_dataset size:', len(train_dataset))
    print(f'test_dataset size:', len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,  # FIXME this is only for the huge dataset to prevent OOM
        # num_workers=4,
        drop_last=True)  # NOTE drop_last

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4)

    log_path = os.path.join(config.checkpoints_dir, 'tb_logs')
    log_path = os.path.join(log_path, config.finetune.train_targets)
    os.makedirs(log_path, exist_ok=True)

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

    if cp_load_path is not None:
        pretrain_target_names = []
        for sub_pretrain_targets in pretrain_targets:
            pretrain_target_names += cp_names_to_targets[sub_pretrain_targets]
        pretrain_target_idxs = [idx for tgt,idx
                     in train_loader.dataset.target_idxs.items()
                     if tgt in pretrain_target_names]
        

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

        sd_new = {}
        for p_name, p in state_dict.items():
            if p_name.startswith('heads'):
                head_idx = int(p_name.split('.')[1])
                
                # avoiding copying targets if didn't train on them
                # This is for fixing failed finetuning
                # because of zero weights for other heads
                #  that happened due to weight decay
                if head_idx not in pretrain_target_idxs:
                    continue
            sd_new[p_name] = p
        state_dict = sd_new
        del loaded_dict
    else:
        state_dict = model.state_dict()

    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.train()

    # if not hasattr(config.finetune, 'pretrained_weights'):
    #     if config.pre_type is None:
    #         config.finetune.pretrained_weights = None
    #     else:
    #         config.finetune.pretrained_weights = \
    #         f"checkpoints/finetune/{config.pretrains[config.pre_type][1]}"
    #         # f"checkpoints/pretrain/{config.pretrains[config.pre_type][1]}"

    # if config.finetune.pretrained_weights is not None:
    #     state_dict = load_pretrained_weights(model, config)

    #     # TODO add flag to configs: use_pretrained_stats
    #     loaded_state_dict = torch.load(
    #         config.finetune.pretrained_weights,
    #         map_location=torch.device(config.device_index))
        
    #     dataset_mean = loaded_state_dict['mean']
    #     dataset_std = loaded_state_dict['std']
    #     print(f'loaded mean {dataset_mean:.2f} and std {dataset_std:.2f}')
    #     del loaded_state_dict
    # else:
    #     state_dict = model.state_dict()
    # if not torch.cuda.is_available():
    #     raise RuntimeError('no gpu')
    # else:
    #     device = torch.device(f'cuda:{config.device_index}')
    #     model.to(device)
    # model.load_state_dict(state_dict, strict=False)
    # model.train()

    optimizer = torch.optim.RAdam(
        model.parameters(), **config.finetune.optimizer_params)
    # optimizer = RAdam(model.parameters(), **config.finetune.optimizer_params)
    # optimizer = torch.optim.SGD(model.parameters(), **config.finetune.optimizer_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, **config.finetune.scheduler_params)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=config.finetune.optimizer_params['lr'],
    #     total_steps=config.epochs,
    # )

    cp_save_prefix = os.path.join(config.checkpoints_dir,
                                  f'{config.finetune.train_targets}')
    cp_save_path = f'{cp_save_prefix}_last.pth'
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.finetune.epochs, eta_min=1e-6)
    # new_results_table, new_eval_table = train_on_split(
    # save_frequency = 1  # in epochs
    preds_df = train_on_subset(
        train_loader=train_loader,
        test_loader=test_loader,
        log_path=log_path,
        scheduler=scheduler,
        config=config,
        device=device,
        dataset_means=dataset_means,
        dataset_stds=dataset_stds,
        model=model,
        optimizer=optimizer,
        # target_name=target_name,
        target_names=target_names,
        save_preds=save_preds,
        cp_save_prefix=cp_save_prefix,
        )
    
    if save_preds:
        preds_save_path = (f'{config.checkpoints_dir}/finetune/'
                            f'{config.name}_final.csv')
        preds_df.to_csv(preds_save_path, index=False)

    
    # cp_save_path = (f'{config.checkpoints_dir}/finetune/'
    #                 f'{config.name}_simple_subset_'
    #                 f'{len(train_dataset)}_last.pth')
    saved_means = {tgt_name:mean for tgt_name,mean
                   in dataset_means.items()
                   if tgt_name in target_names}
    saved_stds = {tgt_name:std for tgt_name,std
                  in dataset_stds.items()
                  if tgt_name in target_names}
    assert saved_means
    assert saved_stds
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': saved_means,
        'std': saved_stds,
        }, cp_save_path)
    t2 = time.time()
    training_time_in_hours = (t2-t1)/60/60
    print(f'"{config.finetune.train_targets}" finished.'
          f' Training time: {training_time_in_hours:.2f} hours.', flush=True)

    
# def train_on_split(train_dataset, val_dataset, test_dataset,
def train_on_subset(*, train_loader, test_loader, log_path, scheduler,
                    config, device, dataset_means, dataset_stds,
                    model, optimizer, target_names, save_preds,
                    cp_save_prefix):
    writer = SummaryWriter(log_path)
    
    # progressbar = tqdm(range(1, config.finetune.epochs+1))
    # progressbar = tqdm(range(1, config.finetune.epochs+1),
    #                     desc=f"Subset size {config.finetune.subset_size}")
    best_test_MAE = float('inf')
    global_batch_count = 0

    # print('train_loader.dataset.target_idxs', train_loader.dataset.target_idxs)
    train_targets = {tgt:idx for tgt,idx
                     in train_loader.dataset.target_idxs.items()
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
    # print('train_perm_targets_mask', train_perm_targets_mask)
    
    # TODO remove?
    # gauss_means = torch.zeros([config.batch_size])
    # gauss_stds = torch.full([config.batch_size], fill_value=40)
    # grad_norms = {}

    best_test_loss = None
    best_test_RMSE = None
    best_test_R2 = None
    best_test_MAE = None

    # for epoch in progressbar:
    for epoch in range(1, config.finetune.epochs+1):
        # t1 = time.time()
        model.train()
        # train_loss = 0.0
        train_loss = torch.zeros(len(target_names), device=device)
        # train_MAE = 0.0

        # cp_save_path_prefix = os.path.join(config.checkpoints_dir,
        #                                    config.finetune.train_targets)
        # cp_save_path_prefix = (f'{config.checkpoints_dir}/finetune/'
        #                        f'{config.name}_simple_subset_'
        #                        f'{len(train_loader.dataset)}_epoch_{epoch}')
        for batch_idx, data in enumerate(train_loader):
            global_batch_count += 1

            # TODO remove? noisy labels
            # target = (data.y + torch.normal(gauss_means, gauss_stds)).to(device)

            data = data.to(device)
            
            # range_mask = (data.y>=config.min_t)&(data.y<=config.max_t)  # NOTE for Tg only!
            # target = (target - dataset_mean)/dataset_std
            
            # target = data.y[:, target_idxs_tensor]
            target = data.y.clone()[:, train_target_idxs_mask]
            # target = data.y[:, train_loader.dataset.target_idxs]
            if train_perm_targets:
                target[:, train_perm_targets_mask] =\
                      torch.log(target[:, train_perm_targets_mask]+1)
            # target = torch.log(target+1)
            target = (target - dataset_means_tensor)/dataset_stds_tensor
            target = target.float()  # double -> float
            # target = (target - dataset_mean)/dataset_std
            # target = (data.y - dataset_mean)/dataset_std
            # target = target[range_mask]
            # real_batch_size = len(target)

            # target = torch.zeros_like(data.y)
            # tgs = torch.split(data.tgs, data.num_tgs.tolist())
            # for mol_idx, mol_tgs in enumerate(tgs):
            #     target[mol_idx] = random.choice(mol_tgs)
            
            pred = model(data).squeeze()

            pred = pred[:, train_target_idxs_mask]
            # pred = pred[range_mask]

            
            # loss = F.l1_loss(pred.squeeze(), target, reduction='none')
            # loss = F.smooth_l1_loss(pred.squeeze(), target, beta=1.0, reduction='mean')
            loss = F.mse_loss(pred, target, reduction='none')  # FIXME divide by num batches if accumulating?
            # train_loss += loss.detach().sum().item()

            with torch.no_grad():
                batch_loss = loss.detach().sum(dim=0)
            train_loss += batch_loss
            # train_MAE += F.l1_loss(pred.squeeze(), target, reduction='sum').item()
            
            mean_loss = loss.mean()
            optimizer.zero_grad()
            mean_loss.backward()

            # if epoch == 1:
            #     for param_name, param in model.named_parameters():
            #         if not param.requires_grad:
            #             grad_norm = -1
            #         elif param.grad is None:
            #             grad_norm = -2
            #         else:
            #             grad_norm = param.grad.detach().data.norm(2).item()
            #         grad_norms.setdefault(param_name, []).append(grad_norm)


            # if global_batch_count % 4 == 0:  # 16 * 4 = 64
            optimizer.step()
            # optimizer.zero_grad()

            # with torch.no_grad():

            #     pred_denormalized = pred*dataset_stds_tensor+dataset_means_tensor
            #     if train_perm_targets:
            #         pred_denormalized[:, train_perm_targets_mask] =\
            #             torch.exp(pred_denormalized[:, train_perm_targets_mask])-1
            #     # pred_denormalized = torch.exp(pred_denormalized)-1
            #     # target_original = data.y[:, train_loader.dataset.target_idx]
            #     target_original = data.y.clone()[:, train_target_idxs_mask]

            #     # NOTE reduction?
            #     batch_MAE = F.l1_loss(pred_denormalized, target_original, reduction='sum').item()
            #     # batch_MAE = F.l1_loss(pred, target, reduction='sum').item()
            # train_MAE += batch_MAE
            # logged_dict = {
            #     f"MSE_{tgt_name}": float(batch_loss[tgt_idx])
            #     for tgt_idx, tgt_name in enumerate(train_targets.keys())
            # }

            # logged_dict = {
            #     "MSE": mean_loss.item(),
            #     # "MAE": batch_MAE/config.batch_size,
            #     # "MAE": batch_MAE/real_batch_size,
            #     # "denorm. MAE": batch_MAE/config.batch_size*dataset_std.item(),
            #     # "denorm. MAE": batch_MAE/real_batch_size*dataset_std.item(),
            #     }
            # progressbar.set_postfix(logged_dict)


        # if global_batch_count % config.finetune.val_freq == 0:
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
        # print(f'epoch {epoch:2d}'
        #     f', LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # message = 'test losses: '
        # for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        #     message += f'"{tgt_name}": {float(test_loss[tgt_idx]):.4f}'
        # print(message)

        # message = 'test MAEs: '
        # for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        #     message += f'"{tgt_name}": {float(test_MAE[tgt_idx]):.4f}'
        # print(message)

        for tgt_idx, tgt_name in enumerate(train_targets.keys()):
            writer.add_scalar(f'Loss_MSE_{tgt_name}/test',
                              float(test_loss[tgt_idx]), epoch)
            writer.add_scalar(f'RMSE_{tgt_name}/test',
                              float(test_RMSE[tgt_idx]), epoch)
            writer.add_scalar(f'R2_{tgt_name}/test',
                              float(test_R2[tgt_idx]), epoch)
            writer.add_scalar(f'MAE_denormalized_{tgt_name}/test',
                              float(test_MAE[tgt_idx]), epoch)
            

        # best_test_loss = None
        # best_test_RMSE = None
        # best_test_R2 = None
        # best_test_MAE = None
        if best_test_loss is None:
            best_test_loss = test_loss
        else:
            for tgt_idx in range(len(train_targets)):
                if test_loss[tgt_idx] < best_test_loss[tgt_idx]:
                    best_test_loss[tgt_idx] = test_loss[tgt_idx]

        if best_test_RMSE is None:
            best_test_RMSE = test_RMSE
        else:
            for tgt_idx in range(len(train_targets)):
                if test_RMSE[tgt_idx] < best_test_RMSE[tgt_idx]:
                    best_test_RMSE[tgt_idx] = test_RMSE[tgt_idx]

        if best_test_R2 is None:
            best_test_R2 = test_R2
        else:
            for tgt_idx in range(len(train_targets)):
                if test_R2[tgt_idx] > best_test_R2[tgt_idx]:
                    best_test_R2[tgt_idx] = test_R2[tgt_idx]

        if best_test_MAE is None:
            best_test_MAE = test_MAE
        else:
            for tgt_idx in range(len(train_targets)):
                if test_MAE[tgt_idx] < best_test_MAE[tgt_idx]:
                    best_test_MAE[tgt_idx] = test_MAE[tgt_idx]

        if epoch == config.finetune.epochs:  # training ended
            csv_dict = {}
            # 'test_loss': test_loss
            for tgt_idx, tgt_name in enumerate(train_targets.keys()):
                csv_dict[f'Test_Loss_MSE_{tgt_name}'] = [float(best_test_loss[tgt_idx])]
            for tgt_idx, tgt_name in enumerate(train_targets.keys()):
                csv_dict[f'Test_RMSE_{tgt_name}'] = [float(best_test_RMSE[tgt_idx])]
            for tgt_idx, tgt_name in enumerate(train_targets.keys()):
                csv_dict[f'Test_R2_{tgt_name}'] = [float(best_test_R2[tgt_idx])]
            for tgt_idx, tgt_name in enumerate(train_targets.keys()):
                csv_dict[f'Test_MAE_{tgt_name}'] = [float(best_test_MAE[tgt_idx])]
            csv_df = pd.DataFrame(csv_dict)
            save_path = f'{cp_save_prefix}_best.csv'
            csv_df.to_csv(save_path, index=False)

            
        #     message = config.finetune.train_targets
        #     message += '\ntest losses: '
        #     for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        #         message += f'"{tgt_name}": {float(test_loss[tgt_idx]):.4f} '
        #     # print(message)

        #     message += '\ntest MAEs: '
        #     for tgt_idx, tgt_name in enumerate(train_targets.keys()):
        #         message += f'"{tgt_name}": {float(test_MAE[tgt_idx]):.2f} '
        #     print(message)

        # FIXME comparison
        # if test_MAE < best_test_MAE:
        #     best_test_MAE = test_MAE
        #     cp_save_path = (f'{cp_save_path_prefix}_best.pth')
        #     saved_means = {tgt_name:mean for tgt_name,mean
        #                    in dataset_means
        #                    if tgt_name in target_names}
        #     saved_stds = {tgt_name:std for tgt_name,std
        #                   in dataset_stds
        #                   if tgt_name in target_names}
        #     assert saved_means
        #     assert saved_stds
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'mean': saved_means,
        #         'std': saved_stds,
        #         'test_MAE': test_MAE,
        #         }, cp_save_path)
            
        # cp_save_path = (f'{cp_save_path_prefix}'
        #                 f'_global_batch_{global_batch_count}'
        #                 f'_periodic.pth')
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'mean': dataset_mean,
        #     'std': dataset_std,
        #     }, cp_save_path)
        model.train()

                # test_loss, test_MAE = evaluate(model=model,
                #                             test_loader=test_loader,
                #                             device=device,
                #                             dataset=dataset,
                #                             writer=writer)
                # model.train()
                # if test_MAE < best_test_MAE:
                #     best_test_MAE = test_MAE
                #     cp_save_path = cp_path_prefix+'_best.pt'
                #     save_checkpoint(model=model,
                #                     mean=dataset.mean,
                #                     std=dataset.std,
                #                     cp_save_path=cp_save_path)
                # if global_batch_count % config.pretrain.save_freq == 0:
                #     cp_save_path = cp_path_prefix+'_periodic.pt'
                #     save_checkpoint(model=model,
                #                     mean=dataset.mean,
                #                     std=dataset.std,
                #                     cp_save_path=cp_save_path)
                    
        # optimizer.step()  # NOTE: Gradient accumulation / full batch training
        # optimizer.zero_grad()
        scheduler.step()

        # if epoch == 1:
        #     gradnorms_save_path = (f'{cp_save_path_prefix}'
        #                            f'_global_batch_{global_batch_count}'
        #                            f'_grad_norms.pth')
        #     torch.save({
        #         'grad_norms': grad_norms,
        #         }, gradnorms_save_path)

        mean_epoch_loss = (float(train_loss.sum())
                           / (train_loss.shape[0]
                              * len(train_loader) * config.batch_size))
        writer.add_scalar(f'Loss_MSE_mean/train', mean_epoch_loss, epoch)
        
        train_loss /= (len(train_loader) * config.batch_size)

        
        # assert dataset_std == train_loader.dataset.std
        # train_MAE_denormalized = (train_MAE
        #                           / (len(train_loader) * config.batch_size)
        #                           * dataset_std.item())

        for tgt_idx, tgt_name in enumerate(train_targets.keys()):
            writer.add_scalar(f'Loss_MSE_{tgt_name}/train',
                              float(train_loss[tgt_idx]), epoch)
        # writer.add_scalar(f'MAE/train_denormalized', train_MAE_denormalized, epoch)
        # t2 = time.time()

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

        # NOTE eval every epoch
        # test_loss, test_MAE, return_table = evaluate(
        #     test_loader=test_loader,
        #     save_preds=True,
        #     target_name=target_name,
        #     model=model,
        #     dataset_mean=dataset_mean,
        #     dataset_std=dataset_std,
        #     device=device)
        # print(f'epoch {epoch:2d}'
        #       f', train loss: {train_loss:.4f}'
        #       f', test loss: {test_loss:.4f}'
        #       f', train MAE: {train_MAE_denormalized:.2f}K'
        #       f', test MAE: {test_MAE*dataset_std.item():.2f}K'
        #       f', LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # if epoch % save_frequency == 0:
        #     cp_save_path = (f'{config.checkpoints_dir}/finetune/'
        #             f'{config.name}_simple_subset_'
        #             f'{len(train_loader.dataset)}_epoch_{epoch}.pth')
        #     torch.save({
        #         'model_state_dict': model.state_dict(),
        #         'mean': dataset_mean,
        #         'std': dataset_std,
        #         }, cp_save_path)
    # return_table_eval = None
    # if eval_dataset is not None:
    #     _, _, return_table_eval = evaluate(eval_dataset, split_idx,
    #                                        save_preds=True, logging=False,
    #                                        log_tag='eval', target_name=target_name)
    writer.close()
    return preds_df#, return_table_eval
    

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
            if train_perm_targets:
                pred_denormalized[:, train_perm_targets_mask] =\
                    torch.exp(pred_denormalized[:, train_perm_targets_mask])-1
            # if i == len(test_loader)-1:
            #     print('pred_denormalized after train_perm_targets:')
            #     print(pred_denormalized)
                
            target_original = test_data.y[:, train_target_idxs_mask]
            target_original = target_original.float()  # double -> float
            # if i == len(test_loader)-1:
            #     print('target_original:')
            #     print(target_original)

            target = target_original.clone()
            if train_perm_targets:
                target[:, train_perm_targets_mask] =\
                      torch.log(target[:, train_perm_targets_mask]+1)
            # target = torch.log(target+1)
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
            test_RMSE += F.mse_loss(pred_denormalized, target_original, reduction='none').sum(dim=0)
            # test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()  # TODO uncomment
            # test_MAE += F.l1_loss(pred_denormalized, target_original, reduction='sum').item()
            test_MAE += F.l1_loss(pred_denormalized, target_original, reduction='none').sum(dim=0)

            all_preds.append(pred_denormalized.cpu().numpy())
            all_targets.append(target_original.cpu().numpy())
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

