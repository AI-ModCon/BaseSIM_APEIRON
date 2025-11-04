import subprocess
import torch
import torch.nn as nn

import os


def get_available_device(multi_gpu=False):
    print("multi gpu: ", multi_gpu)
    if multi_gpu:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Get GPU memory usage using nvidia-smi
        cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_used = subprocess.check_output(cmd.split()).decode().strip().split("\n")
        memory_used = [int(memory.strip()) for memory in memory_used]

        # Find GPU with least memory usage
        device = memory_used.index(min(memory_used))
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"

        return torch.device(f"cuda:{device}")
