# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()
pretrains = {
        "100000b": ("100000", "pretrain_100000Fri Sep 17 18 14 38 2021_best.pt"),
        }

# pre_type = None
pre_type = '100000b'

epochs = 60
# epochs = 30  # no pretrain

dataset_path = "/polymers/GCNN_PI_glass_transition/datasets/PI_exp"
finetune_dataset_name = dataset_path.split('/')[-1]

# eval_dataset_path = "/polymers/GCNN_PI_glass_transition/datasets/PI_exp"

if pre_type is None:
    base_name = f"finetune without pretrain "
else:
    base_name = f"finetuning on {finetune_dataset_name} weights pretrained on {pre_type}"

name = base_name + "#" + runtime
checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 15
seed = 12
device_index = 9

from models import KGNNModel
model = KGNNModel()


# PRETRAIN CONFIG
pretrain = Config(False)
if pretrain.state:
    from data.datasets import SynteticDataset
    pretrain.dataset = SynteticDataset()
    pretrain.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-8}
    pretrain.sheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}
    pretrain.val_freq = 10000
    pretrain.save_freq = 1000
    pretrain.epochs = 20


# FINETUNE CONFIG
finetune = Config(True)
if finetune.state:
    from data.datasets import ExperimentalDataset

    finetune.eval = False
    # finetune.eval = True

    # finetune.eval_dataset = ExperimentalDataset(root=eval_dataset_path)
    finetune.dataset = ExperimentalDataset(root=dataset_path)

    finetune.epochs = epochs
    finetune.n_splits = 10
    if pre_type is None:
        finetune.pretrained_weights = None
    else:
        finetune.pretrained_weights = f"checkpoints/pretrain/{pretrains[pre_type][1]}"
    finetune.optimizer_params = {"lr": 6e-4, "weight_decay": 1e-8}
    finetune.sheduler_params = {"milestones": [50], "gamma": 0.5}
    pass
