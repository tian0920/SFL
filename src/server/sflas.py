from argparse import ArgumentParser, Namespace
from typing import Any, Dict
from copy import deepcopy

from omegaconf import DictConfig
from src.utils.trainer import FLbenchTrainer
from src.utils.constants import MODE

from src.client.sflas import SFLASClient
from src.server.fedavg import FedAvgServer


class SFLASServer(FedAvgServer):
    algorithm_name: str = "sflas"
    return_diff = False
    client_cls = SFLASClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--alignment_lr", type=float, default=0.01)
        parser.add_argument("--alignment_epoch", type=int, default=1)
        return parser.parse_args(args_list)

    def __init__(self, args: DictConfig):
        args.mode = "per_sequential"
        super().__init__(args)
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}

    def init_trainer(self, **extras):
        if self.args.mode == "per_sequential":
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.PER_SEQUENTIAL,
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer_cls(),
                    lr_scheduler_cls=self.get_client_lr_scheduler_cls(),
                    args=self.args,
                    dataset=self.dataset,
                    data_indices=self.client_data_indices,
                    device=self.device,
                    return_diff=self.return_diff,
                    **extras,
                ),
            )

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""
        models = self.trainer.train()
        for client_id, package in models.items():
            self.client_prev_model_states[client_id] = package["prev_model_state"]

        last_item = models.popitem(last=True)
        self.model.load_state_dict(last_item[1]['regular_model_params'], strict=False)

    def package(self, client_id: int):
        server_package = super().package(client_id)
        server_package["current_epoch"] = self.current_epoch
        if client_id in self.client_prev_model_states:
            server_package["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            server_package["prev_model_state"] = None
        return server_package