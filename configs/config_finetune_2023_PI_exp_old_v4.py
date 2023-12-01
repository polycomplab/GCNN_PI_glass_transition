# This is a base example of config file
class Config:
    def __init__(self, state):
        self.state = state
# GENERAL
import time
runtime = time.ctime()
pretrains = {
        # "mixed": ("mixed 5000", "pretrain_mixed_5000Wed Oct  6 19:48:51 2021_best.pt"),
        # "1000b": ("1000", "pretrain_1000Wed Sep 15 17:39:48 2021_best.pt"),  # not found in folder
        # "5000b": ("5000", "pretrain_5000Thu Sep 16 13:38:02 2021_best.pt"),  # not found in folder
        # "100000b": ("100000", "pretrain_100000Fri Sep 17 18 14 38 2021_best.pt"),

        # '1000_old': ('1000_old', 'config187_ModifiedGatedGraphConv_3Convs_3Layers_23-Apr-2021_17:07:07_452883_pretrain_Askadsky_1000_best.pt'),
        # '5000_old': ('5000_old', 'config187_ModifiedGatedGraphConv_3Convs_3Layers_04-Apr-2021_08:19:54_319973_pretrain_Askadsky_5000_best.pt'),
        # '100K_old': ('100K_old', 'config187_ModifiedGatedGraphConv_3Convs_3Layers_15-Apr-2021_11:26:00_147224_pretrain_Askadsky_100000_best.pt'),

        '1000_just_trained': ('1000_just_trained', 'pretrain_1000_PI_syn_Tg, K_Wed Sep 27 13:11:22 2023_best_current_copy.pt'),
        '5000_just_trained': ('5000_just_trained', 'pretrain_5000_PI_syn_Tg, K_Wed Sep 27 13:12:05 2023_best_current_copy.pt'),
        '5000_just_trained_v2': ('5000_just_trained_v2', 'pretrain_5000_PI_syn_Tg, K_Wed Sep 27 13:12:05 2023_best_current_copy_v2.pt'),
        '100K_just_trained': ('100K_just_trained', 'pretrain_100000_PI_syn_Tg, K_Thu Sep 28 00:05:51 2023_best_current_copy.pt'),
        '100K_just_trained_v2': ('100K_just_trained', 'pretrain_100000_PI_syn_Tg, K_Thu Sep 28 00:05:51 2023_best_current_copy_v2.pt'),

        # "1000000p": ("1000000 periodic", "pretrain_1000000Fri Oct  8 18:13:37 2021_periodic.pt"),
        # "1000000b": ("1000000", "pretrain_1000000Fri Oct  8 18:13:37 2021_best.pt"),
        # "PA_5000b": ("5000 PA", "pretrain_5000Fri Nov  4 05:33:51 2022_best.pt"),
        # 'PA_fullb': ('full PA', "pretrain_fullFri Nov  4 09:42:49 2022_best.pt"),
        # 'PA_100000b': ('100000 PA', 'pretrain_100000Sun Nov  6 22:33:09 2022_best.pt'),
        
        # "new_smth": ("new", "pretrain_1000000Mon May 16 13:01:47 2022_best.pt"),
        # "new_mixed_100000": ("new_mixed_100000", "pretrain_mixed_64000Mon May 16 19:09:38 2022_best.pt"),
        # "new_64000": ("new_64000", "pretrain_64000Mon May 16 13:59:03 2022_periodic.pt"),
        # "new_all":("new_all", "pretrain_6000000Tue May 17 01:12:51 2022_best.pt"),
        # "new_all_last":("new_all_last", "pretrain_6000000Tue May 17 01:12:51 2022_periodic.pt"),
        # "1000": ("1000", "pretrain_1000Fri Jul  1 11:57:42 2022_best.pt"),
        # "500": ("500", "pretrain_500Fri Jul  1 19:53:49 2022_best.pt"),
        # "100": ("100", "pretrain_100Fri Jul  1 19:09:21 2022_final.pt"),
        # "3000": ("3000", "pretrain_3000Fri Jul  1 12:15:35 2022_best.pt"),
        # "all_short_new": ("all_short_new", "pretrain_short_allFri Jul  1 14:51:21 2022_best.pt"),
        }
# pre_type = "PA_fullb"
# pre_type = 'PA_100000b'
# pre_type = 'PA_5000b'
# pre_type = '1000b'
# pre_type = '1000_old'
# pre_type = '1000_just_trained'
# pre_type = '5000_old'
# pre_type = '5000_just_trained'
# pre_type = '5000_just_trained_v2'
# pre_type = '100K_just_trained'
# pre_type = '100K_old'
# pre_type = '100000b'
# pre_type = None

# pre_type = '100000b'

dropout_p = 0.1
# dropout_p = 0.9

# epochs = 60  # TODO
# epochs = 100
# epochs = 30  # no pretrain
# epochs = 200
# epochs = 300

# prefix = 'round_10_new_db_'
# prefix = 'round_10_db_PI_10_11_2022_without_atoms_P_Na_and_with_3_endpoints_'
prefix = 'round_31_really_old_PI_exp_db'

# dataset_path = "datasets/PA_exp"
dataset_path = "datasets/PI_exp"
# dataset_path = "datasets/PI_exp_new_10_11_2022"
# dataset_path = "datasets/PI_exp_new_07_03_2023"
# dataset_path = "datasets/PI_exp_new_07_03_2023_multi_target"
finetune_dataset_name = dataset_path.split('/')[-1]

eval_dataset_path = "datasets/PI_exp"
# eval_dataset_path = "datasets/PI_exp_new_10_11_2022"
# eval_dataset_path = "datasets/PI_exp_new_07_03_2023"


# if pre_type is None:
#     base_name = f"finetuning on {finetune_dataset_name} without pretrain "
# else:
#     base_name = f"finetuning on {finetune_dataset_name} weights pretrained on {pre_type}"
base_name = f"finetuning on {finetune_dataset_name}"

name = prefix + base_name + "#" + runtime
checkpoints_dir = "checkpoints"
log_dir = "logs"
batch_size = 64
seed = 12
# device_index = 6

from models import KGNNModel
model = KGNNModel(dropout_p=dropout_p)


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
    # finetune.eval = True  # NOTE with eval!!!!

    finetune.target_name = "Tg, K"
    # finetune.eval_dataset = ExperimentalDataset(root=eval_dataset_path, target_name=finetune.target_name)   # NOTE with eval!!!!
    finetune.dataset = ExperimentalDataset(root=dataset_path, target_name=finetune.target_name)

    # finetune.epochs = epochs
    finetune.n_splits = 10
    # if pre_type is None:
    #     finetune.pretrained_weights = None
    # else:
    #     finetune.pretrained_weights = f"checkpoints/pretrain/{pretrains[pre_type][1]}"
    finetune.optimizer_params = {"lr": 1e-4, "weight_decay": 1e-8}
    # finetune.optimizer_params = {"lr": 6e-4, "weight_decay": 1e-8}
