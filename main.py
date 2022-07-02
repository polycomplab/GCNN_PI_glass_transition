import argparse
import importlib.util
from scripts.pretrain import pretrain
from scripts.finetune import finetune

def main(args):
    spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    if config.pretrain.state:
        pretrain(config)

    if config.finetune.state:
        finetune(config)
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument('--resume', dest='resume', action='store_true')
    args = parser.parse_args()
    main(args)
