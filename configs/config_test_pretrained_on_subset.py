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

# base_name = f"pretrain_{subset}" if subset is not None else f'pretrain_full'
# base_name += f'_{dataset_name}_{target_name}_'
# name = base_name + runtime


checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 20  # effective batch size is this * 4 due to gradients accumulation
seed = 12
device_index = 6

from models import KGNNModel
model = KGNNModel()


# PRETRAIN CONFIG
pretrain = Config(True)
if pretrain.state:
    from data.datasets import SynteticDataset
    pretrain.dataset = SynteticDataset(
        root=dataset_path,
        target_name=target_name)

    pretrain.only_test_pretrained = True
    pretrain.test_subset_size = 6400
    # pretrain.pretrained_weights = "checkpoints/pretrain/pretrain_1000_PI_syn_Tg, K_Wed Sep 27 13:11:22 2023_best.pt"
    # pretrain.pretrained_weights = "checkpoints/pretrain/pretrain_5000_PI_syn_Tg, K_Wed Sep 27 13:12:05 2023_best.pt"

    # pretrain.pretrained_weights = "checkpoints/pretrain/config187_ModifiedGatedGraphConv_3Convs_3Layers_15-Apr-2021_11:26:00_147224_pretrain_Askadsky_100000_best.pt"  # 5 atom types (incompatible with the current 10 atom types networks)
    # pretrain.pretrained_weights = "checkpoints/pretrain/pretrain_100000Fri Sep 17 18 14 38 2021_best.pt"
    # pretrain.pretrained_weights = "checkpoints/pretrain/pretrain_100000Sun Nov  6 22:33:09 2022_best.pt"
    # pretrain.pretrained_weights =  'checkpoints/pretrain/pretrain_5000_PI_syn_Tg, K_Wed Sep 27 13:12:05 2023_best_current_copy.pt'
    pretrain.pretrained_weights =  'checkpoints/finetune/copy_pretrained_on_100K_PI_syn_2_epochs/100_epochs_lr_0.0006_subset_size_100000_round_66_PI_syn_subset_100Kfinetuning on PI_syn_subset_100K#Sun Oct 29 22:11:24 2023_simple_subset_93600_epoch_2.pth'
   

    # pretrain.optimizer_params = {"lr": 1e-3, "weight_decay": 1e-8}
    # # pretrain.optimizer_params = {"lr": 4e-3, "weight_decay": 1e-8}
    # # pretrain.scheduler_params = {"mode": 'min', "factor": 0.9, "patience": 5}
    # pretrain.scheduler_params = {"mode": 'min', "factor": 0.5, "patience": 0}
    # pretrain.val_freq = 250
    # pretrain.epochs = 500
    # total_num_batches = (subset * pretrain.epochs) // batch_size
    # # pretrain.val_freq = 1000
    # pretrain.val_freq = total_num_batches // 10
    # # pretrain.save_freq = 250
    # # pretrain.save_freq = 1000
    # pretrain.save_freq = pretrain.val_freq
    # # pretrain.epochs = 20
    # # pretrain.epochs = 100
    # pretrain.num_batches_acc = 4  # number of batches to accumulate gradient over
    # pretrain.subset = subset


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
