# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()
pretrains = {
        # "full_b": ("full", "pretrain_fullMon Aug  7 15:52:30 2023_best.pt"),
        # "100000b": ("100000", "pretrain_100000Mon Aug  7 15:51:45 2023_best.pt"),
        # "full_b": ("full", "pretrain_fullMon Aug  7 15:52:30 2023_best.pt"),
        # 'PA_fullb': ('full PA', "pretrain_fullFri Nov  4 09:42:49 2022_best.pt"),
        }

pre_type = None
# pre_type = "PA_fullb"
# pre_type = '100000b'

# epochs = 60
epochs = 200
# epochs = 30  # no pretrain

dataset1_path = "datasets/PA_exp_perm_He"
dataset2_path = "datasets/PA_exp"
finetune_dataset1_name = dataset1_path.split('/')[-1]
finetune_dataset2_name = dataset2_path.split('/')[-1]

# eval_dataset_path = "datasets/PA_exp_perm_He"

if pre_type is None:
    base_name = f"2 targets: finetune without pretrain "
else:
    base_name = f"2 targets: finetuning weights pretrained on {pre_type}"

name = base_name + "#" + runtime
name = "trainloop_v1_" + name  # общий беквард для двух форвардов с объединённым лоссом
# name = "pretrained_perm_full_" + name
# name = "pretrained_PA_full_" + name
# name = "trainloop_v2_" + name  # новая схема форварда - беквард после каждого форварда
# name = "3_task_specific_layers_" + name
name = "2_task_specific_layers_" + name
# name = "loss2_weight_testing_" + name
# name = "huber_loss" + name
# name = "SGD_" + name
# name = "perm_only_" + name
name = "fixed_" + name
# name = "acc_batch_64_" + name  # accumulated gradient: 16 * 4 = 64 (effective batch size)
checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 16
seed = 12
device_index = 6

from models import KGNNModel
model = KGNNModel(num_targets=2, num_task_specific_layers=2)


# PRETRAIN CONFIG
pretrain = Config(False)
if pretrain.state:
    from data.datasets import SynteticDataset
    pretrain.dataset = SynteticDataset()
    pretrain.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-8}
    pretrain.scheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}
    pretrain.val_freq = 10000
    pretrain.save_freq = 1000
    pretrain.epochs = 20


# FINETUNE CONFIG
finetune = Config(True)
if finetune.state:
    from data.datasets import ExperimentalDataset

    finetune.eval = False
    # finetune.eval = True

    finetune.target1_name = 'He, Barrer'  # TODO group dataset path and target name
    finetune.target2_name = 'Tg, K'
    # finetune.eval_dataset = ExperimentalDataset(root=eval_dataset_path, target_name=finetune.target_name)
    finetune.dataset1 = ExperimentalDataset(root=dataset1_path, target_name=finetune.target1_name)
    finetune.dataset2 = ExperimentalDataset(root=dataset2_path, target_name=finetune.target2_name)

    finetune.loss1_weight = 1.0
    finetune.loss2_weight = 0.5

    finetune.epochs = epochs
    finetune.n_splits = 10
    if pre_type is None:
        finetune.pretrained_weights = None
    else:
        finetune.pretrained_weights = f"checkpoints/pretrain/{pretrains[pre_type][1]}"
    finetune.optimizer_params = {"lr": 6e-4, "weight_decay": 1e-8}
    # finetune.optimizer_params = {"lr": 6e-4, "momentum":0.9, "weight_decay": 1e-8}
