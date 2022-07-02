import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#from radam import RAdam
from data.KGNNDataLoader import DataLoader
from data.datasets import k_fold_split
import pandas as pd


def finetune(config):
    torch.manual_seed(config.seed)
    writer = SummaryWriter(config.log_dir + "/finetune/" + config.name)

    dataset = config.finetune.dataset
    if config.finetune.eval:
        eval_dataset = config.finetune.eval_dataset
        eval_dataset.std = dataset.std
        eval_dataset.mean = dataset.mean
    else:
        eval_dataset = None
    model = config.model
    if config.finetune.pretrained_weights is not None:
        state_dict = torch.load(config.finetune.pretrained_weights, map_location=torch.device(config.device_index))
    else: 
        state_dict = model.state_dict()
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)

    def evaluate(test_dataset, i, last=False, loss=True):
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        model.eval()
        test_loss = 0.
        test_MAE = 0.
        val_progressbar = tqdm(test_loader, desc=f"Validation")
        data_format = {"ID": [], "Tg, pred": []}
        return_table = pd.DataFrame(data_format)
        with torch.no_grad():
            for i, test_data in enumerate(val_progressbar):
                test_data = test_data.to(device)
                pred = model(test_data).squeeze()
                if last:
                    data_format = {"ID": test_data.index.cpu().numpy().tolist(), "Tg, pred": (pred*dataset.std.item()+dataset.mean.item()).cpu().numpy().tolist()}
                    print(data_format)
                    return_table = return_table.append(pd.DataFrame(data_format), ignore_index=True)
                
                if loss:
                    test_loss += F.mse_loss(pred, test_data.y, reduction='sum').item()
                    test_MAE += F.l1_loss(pred, test_data.y, reduction='sum').item()

        if loss:
            test_loss /= len(test_loader.dataset)
            test_MAE = test_MAE / len(test_loader.dataset)
            print(f"Validation MSE: {test_loss}, MAE: {test_MAE}, denorm. MAE: {test_MAE*dataset.std.item()}")
            writer.add_scalar('Loss_MSE/test', test_loss, i)
            writer.add_scalar('MAE/test', test_MAE, i)
            writer.add_scalar('MAE/test_denormalized', test_MAE*dataset.std.item(), i)
        model.train()
        return test_loss, test_MAE, return_table

    def train_on_split(train_dataset, i, eval_dataset=None):
        print(len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        progressbar = tqdm(range(config.finetune.epochs), desc=f"Training split {i}")
        best_val_MAE = float('inf')
        for epoch in progressbar:
            for data in train_loader:
                data = data.to(device)
                target = data.y

                pred = model(data)
                loss = F.mse_loss(pred.squeeze(), target, reduction='mean')
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            val_loss, val_MAE, _ = evaluate(val_dataset, i)
            if val_MAE < best_val_MAE:
                best_val_MAE = val_MAE
                test_loss, test_MAE, return_table = evaluate(test_dataset, i, last=True)

            #test_loss, test_MAE, return_table = evaluate(test_dataset, i, last=True)
            loss_dict = {
                    "MSE": val_loss,
                    "MAE": val_MAE,
                    "denorm. MAE": (val_MAE*dataset.std),
                    "denorm. test MAE": (test_MAE*dataset.std)
                    }
            progressbar.set_postfix(loss_dict)
        test_loss, test_MAE, return_table = evaluate(test_dataset, i, last=True)
        return_table_eval = None
        if eval_dataset is not None:
            _, _, return_table_eval = evaluate(eval_dataset, i, last=True, loss=False)
        return return_table, return_table_eval
            
    data_format = {"ID": [], "Tg, pred": []}
    results_table = pd.DataFrame(data_format)
    eval_format = {"ID": [], "Tg, pred": []}
    eval_tables = []
    for i, (train_dataset, val_dataset, test_dataset) in enumerate(k_fold_split(dataset, config.finetune.n_splits)):
        print(f"SPLIT {i}")
        model.load_state_dict(state_dict)
        model.train()
        optimizer = torch.optim.RAdam(model.parameters(), **config.finetune.optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config.finetune.sheduler_params)
        new_results_table, new_eval_table = train_on_split(train_dataset, i, eval_dataset)
        results_table = results_table.append(new_results_table, ignore_index = True)
        if config.finetune.eval:
            eval_tables.append(new_eval_table)
    cp_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_final.csv'
    results_table.to_csv(cp_save_path, index=False)
    eval_save_path = f'{config.checkpoints_dir}/finetune/{config.name}_eval.pth'
    torch.save(eval_tables, eval_save_path)
    writer.close()
    

