import copy
from data.datasets import split_train_val, split_subindex
from data.KGNNDataLoader import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from itertools import islice
import time

global_step = 0
best_test_MAE = float('inf')

def pretrain(config):
    torch.manual_seed(config.seed)

    dataset = config.pretrain.dataset
    train_dataset, test_dataset = split_train_val(dataset, 6400)
    if isinstance(config.pretrain.subset, int):
        train_dataset = split_subindex(train_dataset, config.pretrain.subset)
    model = config.model
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)


    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(),
                                **config.pretrain.optimizer_params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config.pretrain.sheduler_params)
    print("HERE")

    def evaluate():
        global global_step
        model.eval()
        test_loss = 0.
        test_MAE = 0.
        val_progressbar = tqdm(test_loader, desc=f"Validation")
        with torch.no_grad():
            for i, test_data in enumerate(val_progressbar):
                test_data = test_data.to(device)
                pred = model(test_data).squeeze()

                test_loss += F.mse_loss(pred, test_data.y, reduction='sum').item()
                test_MAE += F.l1_loss(pred, test_data.y, reduction='sum').item()

        test_loss /= len(test_loader.dataset)
        test_MAE = test_MAE / len(test_loader.dataset)
        print(f"Validation MSE: {test_loss}, MAE: {test_MAE}, denorm. MAE: {test_MAE*dataset.std.item()}")
        writer.add_scalar('Loss_MSE/test', test_loss, global_step)
        writer.add_scalar('MAE/test', test_MAE, global_step)
        writer.add_scalar('MAE/test_denormalized', test_MAE*dataset.std.item(), global_step)
        return test_loss, test_MAE

    def train_epoch():
        global global_step
        global best_test_MAE
        model.train()
        prev_iters = (epoch - 1) * len(train_loader)
        progressbar = tqdm(islice(train_loader, 100000), desc=f"Train epoch {epoch}")

        for i, data in enumerate(progressbar):
            global_step += 1
            data = data.to(device)
            target = data.y

            pred = model(data).squeeze()
            loss = F.mse_loss(pred, target, reduction='mean')
            loss.backward()
            if global_step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging part
            with torch.no_grad():
                batch_MAE = F.l1_loss(pred, target, reduction='mean')
            loss_dict = {
                    "MSE": loss.item(),
                    "MAE": batch_MAE.item(),
                    "denorm. MAE": (batch_MAE.item()*dataset.std).item()
                    }
            progressbar.set_postfix(loss_dict)
            denomalized_batch_MAE = batch_MAE * dataset.std
            writer.add_scalar('Batch_Loss_MSE/train', loss.item(), global_step)
            writer.add_scalar('Batch_MAE/train', batch_MAE.item(), global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('LR', current_lr, global_step)
            # Validation part
            if global_step > 0 and global_step % config.pretrain.val_freq == 0:
                test_loss, test_MAE = evaluate()
                scheduler.step(test_MAE)
                model.train()
                if test_MAE < best_test_MAE:
                    best_test_MAE = test_MAE
                    cp_save_path = f'{config.checkpoints_dir}/pretrain/{config.name}_best.pt'
                    torch.save(model.state_dict(), cp_save_path)
            if global_step % config.pretrain.save_freq == 0:
                cp_save_path = f'{config.checkpoints_dir}/pretrain/{config.name}_periodic.pt'
                torch.save(model.state_dict(), cp_save_path)


    # train loop
    writer = SummaryWriter(config.log_dir + "/pretrain/" + config.name)
    for epoch in range(1, config.pretrain.epochs + 1):
        print(f'epoch {epoch:03d}')
        train_epoch()

    evaluate()
    cp_save_path = f'{config.checkpoints_dir}/pretrain/{config.name}_final.pt'
    torch.save(model.state_dict(), cp_save_path)
    writer.close()
    print(f'best MAE on test: {best_test_MAE:7.4f}, denormalized: {best_test_MAE * dataset.std.item():7.4f}')
