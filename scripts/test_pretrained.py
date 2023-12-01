import copy
from data.data_splitting import split_train_val, split_subindex
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


def load_pretrained_weights(model, config):
    state_dict = torch.load(
            config.pretrain.pretrained_weights,
            map_location=torch.device(config.device_index))
    model_state_dict = model.state_dict()
    for name, tensor in model_state_dict.items():
        # assert name in state_dict
        if name not in state_dict:
            print(f'weights for layer {name} cannot be loaded because'\
                    ' the layer didn\'t exist in that checkpoint. It will be initialized randomly.')
            continue
        if tensor.shape != state_dict[name].shape:
            print(f'weights for layer {name} will not be loaded because of mismatching shape:')
            print(f'weights shape: {state_dict[name].shape}, required shape: {tensor.shape}')
            state_dict[name] = tensor
    for name in state_dict:
        if name not in model_state_dict:
            print(f'checkpoint has weights for missing layer {name}. It won\'t be loaded')
    return state_dict


def test_pretrained(config):
    torch.manual_seed(config.seed)

    dataset = config.pretrain.dataset
    test_size = config.pretrain.test_subset_size
    _, test_dataset = split_train_val(dataset, test_size)

    model = config.model
    if not torch.cuda.is_available():
        raise RuntimeError('no gpu')
    else:
        device = torch.device(f'cuda:{config.device_index}')
        model.to(device)

    state_dict = torch.load(
        config.pretrain.pretrained_weights,
        map_location=torch.device(config.device_index))
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=4)

    def evaluate():
        model.eval()
        test_loss = 0.
        test_MAE = 0.
        val_progressbar = tqdm(test_loader, desc=f"Validation")
        with torch.no_grad():
            for i, test_data in enumerate(val_progressbar):
                test_data = test_data.to(device)
                target_normalized = (test_data.y - dataset.mean)/dataset.std
                pred = model(test_data).squeeze()

                test_loss += F.mse_loss(pred, target_normalized, reduction='sum').item()
                test_MAE += F.l1_loss(pred, target_normalized, reduction='sum').item()

        test_loss /= len(test_loader.dataset)
        test_MAE = test_MAE / len(test_loader.dataset)
        print(f"Validation MSE: {test_loss}, MAE: {test_MAE}"
              f", denorm. MAE: {test_MAE*dataset.std.item()}")
        return test_loss, test_MAE

    _, final_test_MAE = evaluate()  # final test evaluation

    print(f'final MAE on test: {final_test_MAE:7.4f}'
          f', denormalized: {final_test_MAE * dataset.std.item():7.4f}')
