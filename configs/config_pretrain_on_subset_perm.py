# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()

subset = None
# subset = 5000
# subset = 100_000

base_name = f"pretrain_{subset}" if subset is not None else f'pretrain_full'
name = base_name + runtime
checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 20  # effective batch size is this * 4 due to gradients accumulation
seed = 12
# device_index = 6
device_index = 7

from models import KGNNModel
model = KGNNModel()


# PRETRAIN CONFIG
pretrain = Config(True)
if pretrain.state:
    from data.datasets import SynteticDataset
    pretrain.dataset = SynteticDataset(
        root="datasets/PA_syn_perm_He",
        target_name='He, Barrer')  # target property name in dataframe
    pretrain.optimizer_params = {"lr": 4e-3, "weight_decay": 1e-8}  # NOTE changed lr from 1e-3 to 4e-3 for accumulated loss
    # to stay the same after fixing grad accumulation scaling by step size, so that we could tune it correctly
    pretrain.scheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}  # NOTE for ReduceLROnPlateau, probably
    # not optimal (second half of training is useless (lr below 1e-5);
    # lr reduces way too frequently. Increase patience. NOTE patience is not in epochs, in val_freq, and optimizes test MAE!!!!)
    pretrain.val_freq = 1000
    # print('val freq in batches:', 2000)
    # pretrain.val_freq = 250  # NOTE in batches (optimizer steps on every 4th (num_batches_acc'th) batch).
    # That is, after every val_freq/num_batches_acc optimizer steps validation is done, and scheduler is stepped
    pretrain.save_freq = 1000  # periodic saving, in batches
    
    pretrain.num_batches_acc = 4  # number of batches to accumulate gradient over
    # pretrain.epochs = 20
    pretrain.epochs = 100
    pretrain.subset = subset
    pretrain.test_subset_size = 6400


# FINETUNE CONFIG
finetune = Config(False)
if finetune.state:
    from data.datasets import ExperimentalDataset
    finetune.dataset = ExperimentalDataset()
    finetune.epochs = 40
    finetune.n_splits = 10
    finetune.pretrained_weights = "checkpoints/pretrain/Base configThu Jul 22 13:54:43 2021_best.pt"
    finetune.optimizer_params = {"lr": 1e-4, "weight_decay": 1e-8}
    finetune.scheduler_params = {"milestones": [50], "gamma": 0.5}
    pass
