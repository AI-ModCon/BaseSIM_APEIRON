# src/model/mnist_cnn_harness.py
import gc
import os
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, ConcatDataset

from model.torch_model_harness import BaseModelHarness
from config.configuration import Config
from evaluation.metrics import accuracy

from matey import Trainer
from matey.utils import setup_dist, YParams


class MATEY_MODEL(BaseModelHarness):

    def __init__(self, cfg: Config, model: nn.Module = None):

        self.eval_metrics = {"accuracy": accuracy, "loss": self.get_criterion()}
        self.higher_is_better = {"accuracy": True, "loss": False}

        params = YParams(os.path.abspath("examples/matey/input.yaml"), "basic_config")
        params.use_ddp = True
        params.use_fsdp = False
        params.pei_debug = False
        params.pei_oneloss = False
        params.pei_filtered = False
        params.pei_minres = False
        params.pei_moduleloss = False
        params.enable_sync = False
        params.profiling = False

        if not hasattr(params, "tokenizer_heads"):
            assert hasattr(params, "patch_size")
            params.tokenizer_heads=[{"head_name": "default",
                                    "patch_size": params.patch_size 
                                    }]
        print(params.tokenizer_heads, flush=True)
        if hasattr(params, "hierarchical"):
            params.hierarchical["fixedupsample"] = False
            params.hierarchical["linearupsample"] = False
            print(params.hierarchical, flush=True)
        # Set up distributed training
        device, world_size, local_rank, global_rank = setup_dist(params)
        print(f"local_rank={local_rank}, global_rank={global_rank}, world_size={world_size}", flush=True)

        # Modify params
        params['batch_size'] =int(params.batch_size//world_size)
        params['startEpoch'] = 0
        expDir = os.path.join(params.exp_dir, args.config, str(args.run_name))

        params['old_exp_dir'] = expDir # I dont remember what this was for but not removing it yet
        params['experiment_dir'] = os.path.abspath(expDir)
        params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
        params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
        params['old_checkpoint_path'] = os.path.join(params.old_exp_dir, 'training_checkpoints/best_ckpt.tar')

        # Have rank 0 check for and/or make directory
        if  global_rank==0:
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'), exist_ok=True)
        if params.use_fsdp:
            params['resuming'] = True if len(glob.glob(os.path.join(params.best_checkpoint_path, "*distcp")))>0 else False
        else:
            params['resuming'] = True if os.path.isfile(params.best_checkpoint_path) else False

        if params.pei_debug:
            params.debug_outdir = os.path.join(expDir, "./debug_outputs/")
            os.makedirs(params.debug_outdir, exist_ok=True)

        params['log_to_screen'] = (global_rank==0) and params['log_to_screen']
        torch.backends.cudnn.benchmark = False

        self.trainer = Trainer(params, global_rank, local_rank, device)
        super().__init__(cfg=cfg, model=self.trainer.model)

    def get_optmizer(self) -> Optimizer:
        return self.trainer.optimizer

    def get_cur_data_loaders(self):
        return self.trainer.train_data_loader, self.trainer.valid_data_loader

    def update_data_stream(self) -> None:
        return

    def get_hist_data_loaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        return self.trainer.train_data_loader, self.trainer.valid_data_loader

    def get_criterion(self):
        return torch.nn.NLLLoss()
