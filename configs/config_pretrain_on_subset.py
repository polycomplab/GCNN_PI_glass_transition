# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()

# dataset_name = 'PA_syn'
dataset_name = 'PI_syn'
dataset_path = f"datasets/{dataset_name}"
target_name = "Tg, K"
# subset = None
# subset = 1000
# subset = 5000
# subset = 100_000
subset = 10_000

base_name = f"pretrain_{subset}" if subset is not None else f'pretrain_full'
base_name += f'_{dataset_name}_{target_name}_'
name = base_name + runtime
prefix = 'round_62'
name = prefix + name
checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 20  # effective batch size is this * 4 due to gradients accumulation
seed = 12
device_index = 9

from models import KGNNModel
model = KGNNModel()


# PRETRAIN CONFIG
pretrain = Config(True)
if pretrain.state:
    from data.datasets import SynteticDataset
    pretrain.dataset = SynteticDataset(
        root=dataset_path,
        target_name=target_name)
    pretrain.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-8}
    # pretrain.optimizer_params = {"lr": 4e-3, "weight_decay": 1e-8}
    # pretrain.scheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}
    pretrain.scheduler_params = {"mode": 'min', "factor": 0.5, "patience": 0}
    # pretrain.val_freq = 250
    # pretrain.epochs = 500
    pretrain.epochs = 20
    total_num_batches = (subset * pretrain.epochs) // batch_size
    # pretrain.val_freq = 1000
    pretrain.val_freq = total_num_batches // 10
    # pretrain.save_freq = 250
    # pretrain.save_freq = 1000
    pretrain.save_freq = pretrain.val_freq
    # pretrain.epochs = 20
    # pretrain.epochs = 100
    pretrain.num_batches_acc = 4  # number of batches to accumulate gradient over
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
    finetune.sheduler_params = {"milestones": [50], "gamma": 0.5}
    pass
