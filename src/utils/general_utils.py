import os
import subprocess
import torch


def get_available_device(multi_gpu: bool = False) -> torch.device:
    """
    Returns a torch.device with sensible fallbacks:
      - CPU-only hosts: 'cpu'
      - CUDA hosts:
          * multi_gpu=True  -> 'cuda' (let caller handle DDP/DataParallel)
          * multi_gpu=False -> choose GPU with most free memory (torch.cuda.mem_get_info)
            (falls back to nvidia-smi if needed; then to cuda:0)
      - Apple Silicon with PyTorch MPS: 'mps' if CUDA is unavailable
    Never raises if nvidia-smi is missing.
    """
    # Prefer CUDA if available
    if torch.cuda.is_available():
        if multi_gpu:
            return torch.device("cuda")

        # Single-GPU selection: prefer PyTorch's mem_get_info (no external deps)
        try:
            n = torch.cuda.device_count()
            free_bytes = []
            for i in range(n):
                # Ensure we query device i
                with torch.cuda.device(i):
                    free_i, _total_i = torch.cuda.mem_get_info()
                free_bytes.append(free_i)

            best = max(range(n), key=lambda i: free_bytes[i])
            return torch.device(f"cuda:{best}")

        except Exception:
            # Fallback to nvidia-smi if mem_get_info is unavailable/problematic
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                )
                rows = [
                    int(x.strip())
                    for x in out.decode().strip().splitlines()
                    if x.strip()
                ]
                best = max(range(len(rows)), key=lambda i: rows[i])
                return torch.device(f"cuda:{best}")
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Last resort: first CUDA device
                return torch.device("cuda:0")

    # CUDA not available: try MPS (Apple), otherwise CPU
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass

    return torch.device("cpu")
