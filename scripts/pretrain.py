
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_splitting import split_train_val, split_subindex
from data.KGNNDataLoader import DataLoader
from scripts.finetune import evaluate


def pretrain(config):
    torch.manual_seed(config.seed)

    dataset = config.pretrain.dataset
    test_size = config.pretrain.test_subset_size
    train_dataset, test_dataset = split_train_val(dataset, test_size)
    if isinstance(config.pretrain.subset, int):
        train_dataset = split_subindex(train_dataset, config.pretrain.subset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True)

    model = config.model
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)


    # optimizer = torch.optim.Adam(model.parameters(),
    #                             **config.pretrain.optimizer_params)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, **config.pretrain.scheduler_params)
    optimizer = torch.optim.RAdam(
        model.parameters(), **config.pretrain.optimizer_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, **config.pretrain.scheduler_params)

    cp_path_prefix = f'{config.checkpoints_dir}/pretrain/{config.name}'
    writer = SummaryWriter(config.log_dir + "/pretrain/" + config.name)
    global_batch_count = 0
    best_test_MAE = float('inf')
    for epoch in range(1, config.pretrain.epochs + 1):
        print(f'epoch {epoch:03d}')
        best_test_MAE, global_batch_count =\
            train_epoch(model=model,
                        epoch=epoch,
                        train_loader=train_loader,
                        device=device,
                        dataset=dataset,
                        config=config,
                        optimizer=optimizer,
                        writer=writer,
                        test_dataset=test_dataset,
                        scheduler=scheduler,
                        cp_path_prefix=cp_path_prefix,
                        best_test_MAE=best_test_MAE,
                        global_batch_count=global_batch_count,
                        )

    # final test evaluation
    eval_results = evaluate(test_dataset,
                            return_preds_df=False,
                            tb_logging=True,
                            tb_writer=writer,
                            tb_iter_idx=global_batch_count,
                            log_tag='val',
                            batch_size=config.batch_size,
                            model=model,
                            tgt_mean_to_denorm_preds=dataset.mean,
                            tgt_std_to_denorm_preds=dataset.std,
                            tgt_mean_to_norm_tgts=dataset.mean,
                            tgt_std_to_norm_tgts=dataset.std,
                            device=device,
                            print_to_console=True)
    _, final_test_MAE, _ = eval_results
    # _, final_test_MAE = evaluate(model=model,
    #                              test_loader=test_loader,
    #                              device=device,
    #                              dataset=dataset,
    #                              writer=writer,
    #                              global_batch_count=global_batch_count)
    cp_save_path = cp_path_prefix+'_final.pt'
    save_checkpoint(model=model,
                    mean=dataset.mean,
                    std=dataset.std,
                    cp_save_path=cp_save_path)
    writer.close()
    print(f'best MAE on test (denormalized): {best_test_MAE:7.4f}')
    print(f'final MAE on test (denormalized): {final_test_MAE:7.4f}')


def train_epoch(*, model, epoch, train_loader, device, dataset, config,
                optimizer, writer, test_dataset, scheduler, cp_path_prefix,
                best_test_MAE, global_batch_count):
    model.train()
    progressbar = tqdm(train_loader, desc=f"Train epoch {epoch}")

    for i, data in enumerate(progressbar):
        global_batch_count += 1
        data = data.to(device)
        target = (data.y - dataset.mean)/dataset.std

        pred = model(data).squeeze()
        loss = (F.mse_loss(pred, target, reduction='mean')
                / config.pretrain.num_batches_acc)
        loss.backward()
        if global_batch_count % config.pretrain.num_batches_acc == 0:
            # accumulating gradients.
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging part
        with torch.no_grad():
            batch_MAE = F.l1_loss(pred, target, reduction='mean')
        denormalized_batch_MAE = (batch_MAE.item()*dataset.std).item()
        logged_dict = {
                "MSE": loss.item()*config.pretrain.num_batches_acc,  # approximation
                "MAE": batch_MAE.item(),
                "denorm. MAE": denormalized_batch_MAE,
                }
        progressbar.set_postfix(logged_dict)
        writer.add_scalar('Batch_Loss_MSE/train', loss.item(), global_batch_count)
        writer.add_scalar('Batch_MAE/train', batch_MAE.item(), global_batch_count)
        writer.add_scalar('Batch_MAE/train_denormalized',
                            denormalized_batch_MAE, global_batch_count)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar('LR', current_lr, global_batch_count)
        # Validation part
        if global_batch_count % config.pretrain.val_freq == 0:
            eval_results = evaluate(test_dataset,
                                    return_preds_df=True,
                                    tb_logging=True,
                                    tb_writer=writer,
                                    tb_iter_idx=global_batch_count,
                                    log_tag='val',
                                    target_name=config.target_name,
                                    batch_size=config.batch_size,
                                    model=model,
                                    tgt_mean_to_denorm_preds=dataset.mean,
                                    tgt_std_to_denorm_preds=dataset.std,
                                    tgt_mean_to_norm_tgts=dataset.mean,
                                    tgt_std_to_norm_tgts=dataset.std,
                                    device=device,
                                    print_to_console=True)
            _, test_MAE, preds_df = eval_results
            results_save_path = (f'{config.checkpoints_dir}/pretrain/'
                                 f'{config.name}_epoch_{epoch}'
                                 f'_global_batch_{global_batch_count}.csv')
            preds_df.to_csv(results_save_path, index=False)
            # test_loss, test_MAE = evaluate(model=model,
            #                                test_loader=test_loader,
            #                                device=device,
            #                                dataset=dataset,
            #                                writer=writer,
            #                                global_batch_count=global_batch_count)
            model.train()
            if test_MAE < best_test_MAE:
                best_test_MAE = test_MAE
                cp_save_path = cp_path_prefix+'_best.pt'
                save_checkpoint(model=model,
                                mean=dataset.mean,
                                std=dataset.std,
                                cp_save_path=cp_save_path)
        if global_batch_count % config.pretrain.save_freq == 0:
            # cp_save_path = cp_path_prefix+'_periodic.pt'
            cp_save_path = (f'{cp_path_prefix}'
                            f'_epoch_{epoch}'
                            f'_global_batch_{global_batch_count}'
                            f'_periodic.pt')
            save_checkpoint(model=model,
                            mean=dataset.mean,
                            std=dataset.std,
                            cp_save_path=cp_save_path)
    scheduler.step()
    return best_test_MAE, global_batch_count


# def evaluate(*, model, test_loader, device, dataset, writer,
#              global_batch_count):
#         model.eval()
#         test_loss = 0.
#         test_MAE = 0.
#         val_progressbar = tqdm(test_loader, desc=f"Validation")
#         with torch.no_grad():
#             for i, test_data in enumerate(val_progressbar):
#                 test_data = test_data.to(device)
#                 target = (test_data.y - dataset.mean)/dataset.std
#                 pred = model(test_data).squeeze()

#                 test_loss += F.mse_loss(pred, target, reduction='sum').item()
#                 test_MAE += F.l1_loss(pred, target, reduction='sum').item()

#         test_loss /= len(test_loader.dataset)
#         test_MAE /= len(test_loader.dataset)
#         print(f"Validation MSE: {test_loss:.4f}, MAE: {test_MAE:.4f}"
#               f", denorm. MAE: {test_MAE*dataset.std.item():.4f}")
#         writer.add_scalar('Loss_MSE/test', test_loss, global_batch_count)
#         writer.add_scalar('MAE/test', test_MAE, global_batch_count)
#         writer.add_scalar('MAE/test_denormalized',
#                           test_MAE*dataset.std.item(), global_batch_count)
#         return test_loss, test_MAE


def save_checkpoint(*, model, mean, std, cp_save_path):
    # torch.save(model.state_dict(), cp_save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': mean,
        'std': std,
        }, cp_save_path)
