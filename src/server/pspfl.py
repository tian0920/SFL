import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.server.fedavg import FedAvgServer
from src.client.pspfl import PSPFLClient

from src.utils.functional import (
    fix_random_seed,
    get_optimal_cuda_device,
)


class PSPFLServer(FedAvgServer):
    client_cls = PSPFLClient
    return_diff = False
    algorithm_name = "PSPFL"
    all_model_params_personalized = False

    def __init__(self, args: DictConfig):
        self.args = args
        super().__init__(args)

        self.device = get_optimal_cuda_device(self.args.common.use_cuda)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_device(self.device)

        fix_random_seed(self.args.common.seed, use_cuda=self.device.type == "cuda")

        norm_map = {
            'global': 'GlobalNorm',
            'layer': 'LayerNorm'
        }
        norm_name = norm_map.get(self.args.pspfl.norm, 'NoNorm')

        ema_name = f"EMAscore({self.args.pspfl.alpha})" if self.args.pspfl.EMA else "NoEMAscore"

        cls_name = f"CLS" if self.args.pspfl.CLS else "ALL"

        if self.args.pspfl.download:
            self.output_dir = (
                    Path("./out")
                    / self.args.method
                    / self.args.dataset.name
                    / f"{self.args.pspfl.type}_{norm_name}_{ema_name}_{cls_name}_IG_{self.args.pspfl.ig_ratio}"
            )

        if self.args.pspfl.upload:
            self.output_dir = (
                    Path("./out")
                    / self.args.method
                    / self.args.dataset.name
                    / f"{self.args.pspfl.type}_{norm_name}_{ema_name}_{cls_name}_IL_{self.args.pspfl.il_ratio}"
            )

        print(self.output_dir)

        if not os.path.isdir(self.output_dir) and (
                self.args.common.save_log
                or self.args.common.save_learning_curve_plot
                or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)