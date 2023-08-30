import argparse
import importlib.util

from scripts.pretrain import pretrain
from scripts.finetune import finetune, update_finetune_config


def main(args):
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if config.pretrain.state:
        pretrain(config)

    if config.finetune.state:
        update_finetune_config(args, config)
        finetune(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    main(args)
