import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from radam import RAdam
from data.KGNNDataLoader import DataLoader
from data.data_splitting import split_train_subset, split_train_val, split_subindex
from utils import load_pretrained_weights


def finetune_simple_on_subset(config):
    torch.manual_seed(config.seed)

    dataset = config.finetune.dataset
    # if config.finetune.eval:
    #     eval_dataset = config.finetune.eval_dataset
    #     eval_dataset.std = dataset.std
    #     eval_dataset.mean = dataset.mean
    # else:
    #     eval_dataset = None

    # print(f'current dataset mean {dataset.mean:.2f} and std {dataset.std:.2f}')

    target_name = config.finetune.target_name  # target chemical property
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

    log_path = config.log_dir + "/finetune/" + config.name +\
        f'_simple_subset_{len(train_dataset)}'


    model = config.model
    # dataset_mean = dataset.mean
    # dataset_std = dataset.std
    dataset_mean = dataset.mean[target_name]
    dataset_std = dataset.std[target_name]
    
    # TODO remove
    # dataset_mean = dataset.real_mean
    # dataset_std = dataset.real_std

    if not hasattr(config.finetune, 'pretrained_weights'):
        if config.pre_type is None:
            config.finetune.pretrained_weights = None
        else:
            config.finetune.pretrained_weights = \
            f"checkpoints/finetune/{config.pretrains[config.pre_type][1]}"
            # f"checkpoints/pretrain/{config.pretrains[config.pre_type][1]}"

    if config.finetune.pretrained_weights is not None:
        state_dict = load_pretrained_weights(model, config)

        # TODO add flag to configs: use_pretrained_stats
        loaded_state_dict = torch.load(
            config.finetune.pretrained_weights,
            map_location=torch.device(config.device_index))
        
        dataset_mean = loaded_state_dict['mean']
        dataset_std = loaded_state_dict['std']
        print(f'loaded mean {dataset_mean:.2f} and std {dataset_std:.2f}')
        del loaded_state_dict
    else:
        state_dict = model.state_dict()
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)
    model.load_state_dict(state_dict, strict=False)
    model.train()

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
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        model=model,
        optimizer=optimizer,
        target_name=target_name,
        save_preds=save_preds,
        )
    if save_preds:
        preds_save_path = (f'{config.checkpoints_dir}/finetune/'
                            f'{config.name}_final.csv')
        preds_df.to_csv(preds_save_path, index=False)

    cp_save_path = (f'{config.checkpoints_dir}/finetune/'
                    f'{config.name}_simple_subset_'
                    f'{len(train_dataset)}_last.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        }, cp_save_path)

    
# def train_on_split(train_dataset, val_dataset, test_dataset,
def train_on_subset(*, train_loader, test_loader, log_path, scheduler,
                    config, device, dataset_mean, dataset_std,
                    model, optimizer, target_name, save_preds):
    writer = SummaryWriter(log_path)
    
    progressbar = tqdm(range(1, config.finetune.epochs+1))
    # progressbar = tqdm(range(1, config.finetune.epochs+1),
    #                     desc=f"Subset size {config.finetune.subset_size}")
    best_test_MAE = float('inf')
    global_batch_count = 0
    
    
    # TODO remove?
    # gauss_means = torch.zeros([config.batch_size])
    # gauss_stds = torch.full([config.batch_size], fill_value=40)
    # grad_norms = {}

    for epoch in progressbar:
        t1 = time.time()
        model.train()
        train_loss = 0.0
        train_MAE = 0.0
        cp_save_path_prefix = (f'{config.checkpoints_dir}/finetune/'
                               f'{config.name}_simple_subset_'
                               f'{len(train_loader.dataset)}_epoch_{epoch}')
        for batch_idx, data in enumerate(train_loader):
            global_batch_count += 1

            # TODO remove? noisy labels
            # target = (data.y + torch.normal(gauss_means, gauss_stds)).to(device)

            data = data.to(device)
            
            # range_mask = (data.y>=config.min_t)&(data.y<=config.max_t)  # NOTE for Tg only!
            # target = (target - dataset_mean)/dataset_std
            
            target = data.y[:, train_loader.dataset.target_idx]
            target = torch.log(target+1)
            target = (target - dataset_mean)/dataset_std
            # target = (data.y - dataset_mean)/dataset_std
            # target = target[range_mask]
            real_batch_size = len(target)

            # target = torch.zeros_like(data.y)
            # tgs = torch.split(data.tgs, data.num_tgs.tolist())
            # for mol_idx, mol_tgs in enumerate(tgs):
            #     target[mol_idx] = random.choice(mol_tgs)
            
            pred = model(data).squeeze()
            # pred = pred[range_mask]
            
            # TODO change back to MSE
            # loss = F.l1_loss(pred.squeeze(), target, reduction='none')
            # loss = F.smooth_l1_loss(pred.squeeze(), target, beta=1.0, reduction='mean')
            loss = F.mse_loss(pred, target, reduction='none')  # FIXME divide by num batches if accumulating?
            train_loss += loss.detach().sum().item()
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

            with torch.no_grad():
                pred_denormalized = pred*dataset_std+dataset_mean
                pred_denormalized = torch.exp(pred_denormalized)-1
                target_original = data.y[:, train_loader.dataset.target_idx]
                batch_MAE = F.l1_loss(pred_denormalized, target_original, reduction='sum').item()
                # batch_MAE = F.l1_loss(pred, target, reduction='sum').item()
            train_MAE += batch_MAE

            logged_dict = {
                "MSE": mean_loss.item(),
                # "MAE": batch_MAE/config.batch_size,
                "MAE": batch_MAE/real_batch_size,
                # "denorm. MAE": batch_MAE/config.batch_size*dataset_std.item(),
                # "denorm. MAE": batch_MAE/real_batch_size*dataset_std.item(),
                }
            progressbar.set_postfix(logged_dict)

            if global_batch_count % config.finetune.val_freq == 0:
                test_loss, test_MAE, preds_df = evaluate(
                    test_loader=test_loader,
                    save_preds=save_preds,
                    target_name=target_name,
                    model=model,
                    dataset_mean=dataset_mean,
                    dataset_std=dataset_std,
                    device=device,
                    config=config)
                print(f'epoch {epoch:2d}'
                    f', test loss: {test_loss:.4f}'
                    f', test MAE: {test_MAE*dataset_std.item():.2f}K'
                    f', LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                if test_MAE < best_test_MAE:
                    best_test_MAE = test_MAE
                    cp_save_path = (f'{cp_save_path_prefix}'
                                    f'_global_batch_{global_batch_count}'
                                    f'_best.pth')
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'mean': dataset_mean,
                        'std': dataset_std,
                        }, cp_save_path)
                    
                cp_save_path = (f'{cp_save_path_prefix}'
                                f'_global_batch_{global_batch_count}'
                                f'_periodic.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'mean': dataset_mean,
                    'std': dataset_std,
                    }, cp_save_path)
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

        train_loss /= (len(train_loader) * config.batch_size)
        # assert dataset_std == train_loader.dataset.std
        train_MAE_denormalized = (train_MAE
                                  / (len(train_loader) * config.batch_size)
                                  * dataset_std.item())
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
    

def evaluate(*, test_loader, save_preds, target_name, model,
             dataset_mean, dataset_std, device, config):
    model.eval()
    test_loss = 0.
    test_MAE = 0.
    # val_progressbar = tqdm(test_loader, desc=f"Validation")
    if save_preds:
        data_format = {"ID": [], f"{target_name}, pred": []}
        preds_df = pd.DataFrame(data_format)
    else:
        preds_df = None
    data_mean = dataset_mean.item()
    data_std = dataset_std.item()
    num_mols = 0
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_data = test_data.to(device)
            pred = model(test_data).squeeze()
            if save_preds:
                mol_ids = test_data.index.cpu().numpy().tolist()
                pred_denormalized = (pred*data_std + data_mean).cpu().numpy().tolist()
                data_batch = {
                    "ID": mol_ids,
                    f"{target_name}, pred": pred_denormalized}  # FIXME add log for perms
                # print(data_batch)
                preds_df = preds_df.append(  # TODO improve speed here
                    pd.DataFrame(data_batch), ignore_index=True)
            
            # range_mask = (test_data.y>=config.min_t)&(test_data.y<=config.max_t)  # NOTE only for Tg
            # pred = pred[range_mask]
            num_mols += len(pred)
            # target = test_data.y[:, ]
            pred_denormalized = pred*dataset_std+dataset_mean
            pred_denormalized = torch.exp(pred_denormalized)-1
            
            target_original = test_data.y[:, test_loader.dataset.target_idx]
            target = (target_original - dataset_mean)/dataset_std
            target = torch.log(target+1)
            # target = torch.log(target+1)
            # target_normalized = (target - data_mean)/data_std
            # target_normalized = target_normalized[range_mask]
            # test_loss += F.smooth_l1_loss(pred, test_data.y, reduction='sum').item()
            # test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()  # TODO uncomment
            test_loss += F.mse_loss(pred, target, reduction='sum').item()
            # test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()  # TODO uncomment
            test_MAE += F.l1_loss(pred_denormalized, target_original, reduction='sum').item()

    # test_loss /= len(test_loader.dataset)
    test_loss /= num_mols
    # test_MAE /= len(test_loader.dataset)
    test_MAE /= num_mols
    test_MAE_denormalized = test_MAE*data_std
    # print(f"Test loss: {test_loss:.4f}, test MAE (Tg, K): {test_MAE_denormalized:.4f}")
    return test_loss, test_MAE, preds_df

