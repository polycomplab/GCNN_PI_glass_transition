# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()

# subset = None
# subset = 5000
subset = 100_000

base_name = f"pretrain_{subset}" if subset is not None else f'pretrain_full'
name = base_name + runtime
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
        root="datasets/PA_syn")
    pretrain.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-8}
    pretrain.sheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}
    pretrain.val_freq = 250
    pretrain.save_freq = 250
    pretrain.epochs = 20
    pretrain.subset = subset


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
