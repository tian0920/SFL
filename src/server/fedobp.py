import json
import os
import pickle
import random
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import DictConfig
from rich.console import Console
from rich.progress import track

from src.server.fedavg import FedAvgServer
from src.client.fedobp import FedOBPClient
from src.utils.constants import (
    FLBENCH_ROOT,
)
from src.utils.metrics import Metrics
from src.utils.models import MODELS, DecoupledModel
from src.utils.functional import (
    fix_random_seed,
    get_optimal_cuda_device,
    initialize_data_loaders,
)


class FedOBPServer(FedAvgServer):
    client_cls = FedOBPClient
    return_diff = False
    algorithm_name = "FedOBP"
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
        norm_name = norm_map.get(self.args.fedobp.norm, 'NoNorm')

        ema_name = f"EMAscore({self.args.fedobp.alpha})" if self.args.fedobp.EMA else "NoEMAscore"

        cls_name = f"CLS" if self.args.fedobp.CLS else "ALL"

        # Directly combine into a complete output directory
        self.output_dir = (
                Path("./out")
                / self.args.method
                / self.args.dataset.name
                / f"{self.args.fedobp.type}_{norm_name}_{ema_name}_{cls_name}_IG_{self.args.fedobp.ig_ratio}"
        )

        print(self.output_dir)

        if not os.path.isdir(self.output_dir) and (
                self.args.common.save_log
                or self.args.common.save_learning_curve_plot
                or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)

        with open(
                FLBENCH_ROOT / "data" / self.args.dataset.name / "args.json", "r"
        ) as f:
            self.args.dataset.update(DictConfig(json.load(f)))

        # Get client party info
        try:
            partition_path = (
                    FLBENCH_ROOT / "data" / self.args.dataset.name / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                self.data_partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {self.args.dataset.name} first.")
        self.train_clients: List[int] = self.data_partition["separation"]["train"]
        self.test_clients: List[int] = self.data_partition["separation"]["test"]
        self.val_clients: List[int] = self.data_partition["separation"]["val"]
        self.client_num: int = self.data_partition["separation"]["total"]

        # Initialize model(s) parameters
        self.model: DecoupledModel = MODELS[self.args.model.name](dataset=self.args.dataset.name,pretrained=self.args.model.use_torchvision_pretrained_weights,)
        self.model.check_and_preprocess(self.args)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        if self.args.model.external_model_weights_path is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.model.external_model_weights_path).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                self.public_model_params.update(
                    torch.load(file_path, map_location="cpu")
                )
            elif not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file path.")
            elif file_path.find(".pt") == -1:
                raise TypeError(f"{file_path} is not a valid .pt file.")

        self.clients_personal_model_params = {i: {} for i in range(self.client_num)}

        if self.args.common.buffers == "local":
            _init_buffers = OrderedDict(self.model.named_buffers())
            for i in range(self.client_num):
                self.clients_personal_model_params[i] = deepcopy(_init_buffers)

        if self.all_model_params_personalized:
            for params_dict in self.clients_personal_model_params.values():
                params_dict.update(deepcopy(self.model.state_dict()))

        self.client_optimizer_states = {i: {} for i in range(self.client_num)}

        self.client_lr_scheduler_states = {i: {} for i in range(self.client_num)}

        self.client_local_epoches: List[int] = [
                                                   self.args.common.local_epoch
                                               ] * self.client_num

        # System heterogeneity (straggler) setting
        if (
                self.args.common.straggler_ratio > 0
                and self.args.common.local_epoch
                > self.args.common.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.common.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.client_local_epoches = [self.args.common.local_epoch] * (
                normal_num
            ) + random.choices(
                range(
                    self.args.common.straggler_min_local_epoch,
                    self.args.common.local_epoch,
                ),
                k=straggler_num,
            )
            random.shuffle(self.client_local_epoches)

        # To ensure all algorithms run through the same client sampling stream.
        # Some algorithms' implicit operations at the client side may
        # disturb the stream if sampling happens at the beginning of each FL round.
        self.client_sample_stream = [
            random.sample(
                self.train_clients,
                max(1, int(self.client_num * self.args.common.join_ratio)),
            )
            for _ in range(self.args.common.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0

        # For controlling behaviors of some specific methods while testing (not used by all methods)
        self.testing = False

        self.client_metrics = {i: {} for i in self.train_clients}
        self.aggregated_client_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        }

        self.verbose = False

        self.test_results: Dict[int, Dict[str, Dict[str, Metrics]]] = {}
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        if self.args.common.monitor is not None:
            self.monitor_window_name_suffix = (
                self.args.dataset.monitor_window_name_suffix
            )

        if self.args.common.monitor == "visdom":
            from visdom import Visdom

            self.viz = Visdom()
        elif self.args.common.monitor == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(log_dir=self.output_dir)

        self.dataset = self.get_dataset()
        self.client_data_indices = self.get_clients_data_indices()

        # Create setup for centralized evaluation
        (
            self.trainloader,
            self.testloader,
            self.valloader,
            self.trainset,
            self.testset,
            self.valset,
        ) = initialize_data_loaders(
            self.dataset, self.client_data_indices, self.args.common.batch_size
        )

    def package(self, client_id: int):
        return dict(
            client_id=client_id,
            current_epoch=self.current_epoch,
            local_epoch=self.client_local_epoches[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.client_optimizer_states[client_id],
            lr_scheduler_state=self.client_lr_scheduler_states[client_id],
            return_diff=self.return_diff,
        )