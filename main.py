import argparse
import importlib.util

from scripts.pretrain import pretrain
from scripts.finetune import finetune, update_finetune_config, eval_only
from scripts.finetune_simple_on_subset import finetune_simple_on_subset
from scripts.finetune_classifier_simple_on_subset import finetune_classifier_simple_on_subset
from scripts.finetune_simple_on_mixed_subset import finetune_simple_on_mixed_subset
from scripts.finetune_simple_pi_exp_on_subset_multitarget import finetune_on_exp_simple_on_subset_multitarget
from scripts.finetune_simple_on_subset_multitarget import finetune_simple_on_subset_multitarget
from scripts.eval_only_simple_on_subset_multitarget import eval_simple_on_subset_multitarget
from scripts.finetune2targets import finetune_on_2_targets
from scripts.test_pretrained import test_pretrained


def main(args):
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if config.pretrain.state:
        if hasattr(config.pretrain, 'only_test_pretrained'):
            test_pretrained(config)
        else:
            pretrain(config)

    if config.finetune.state:
        if hasattr(config, 'multitarget') and config.multitarget:
            print('finetuning multitarget')
            update_finetune_config(args, config)
            if hasattr(config, 'finetune_exp') and config.finetune_exp:
                finetune_on_exp_simple_on_subset_multitarget(config)
            elif hasattr(config, 'eval_only') and config.eval_only:
                eval_simple_on_subset_multitarget(config)
            else:
                finetune_simple_on_subset_multitarget(config)
        elif hasattr(config.finetune, 'dataset2'):
            update_finetune_config(args, config)
            print('finetuning on 2+ targets')
            finetune_on_2_targets(config)  # older mode
        else:
            # assert False
            print('finetuning on 1 target')
            if (hasattr(config.finetune, 'eval')
                and config.finetune.eval
                and not hasattr(config.finetune, 'dataset')):
                eval_only(config)
            # elif config.finetune.subset_size is not None:
            elif hasattr(config.finetune, 'test_subset_size'):
                update_finetune_config(args, config)
                if hasattr(config.finetune, 'test_dataset'):
                    print('finetune_simple_on_mixed_subset')
                    finetune_simple_on_mixed_subset(config)
                else:
                    if hasattr(config, 'num_bins'):
                        print('finetune_simple_on_subset')
                        finetune_classifier_simple_on_subset(config)
                    else:
                        print('finetune_simple_on_subset')
                        finetune_simple_on_subset(config)
            else:
                update_finetune_config(args, config)
                finetune(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--subset_size', type=int)
    # parser.add_argument('--num_task_layers', type=int)
    parser.add_argument('--loss2_weight', type=float)
    parser.add_argument('--pre_type', type=str)
    parser.add_argument('--split_fixed', type=str)
    parser.add_argument('--train_targets', type=str)
    args = parser.parse_args()
    # print(args)
    main(args)
