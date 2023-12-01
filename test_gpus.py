import argparse
import time

import torch


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=int)
    parser.add_argument('--seconds', type=int)
    args = parser.parse_args()

    mem_size = 1_000_000_000
    device = f'cuda:{args.device_idx}'
    t = torch.ones([mem_size], device=device)

    time.sleep(args.seconds)
