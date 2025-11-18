from collections import OrderedDict
from typing import Any, Dict

import torch
from omegaconf import DictConfig
from src.server.fedavg import FedAvgServer
from src.client.fedpdav2 import FedPDAv2Client


class FedPDAv2Server(FedAvgServer):
    """Server 端仅负责
       1. FedAvg 聚合公共权重
       2. 按样本数加权聚合 class prototypes
    """
    algorithm_name = "FedPDAv2"
    client_cls = FedPDAv2Client
    return_diff = False

    def __init__(self, args: DictConfig):
        super().__init__(args)
        self.global_prototypes = {}
        self.client_prev_model_states: Dict[int, Dict[str, Any]] = {}

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at
        server side) in each communication round."""

        client_packages = self.trainer.train()
        for client_id, package in client_packages.items():
            self.client_prev_model_states[client_id] = package["prev_model_state"]
        self.aggregate_client_updates(client_packages)

    def package(self, client_id: int):
        pkg = super().package(client_id)
        pkg["global_prototypes"] = self.global_prototypes
        pkg["current_epoch"] = self.current_epoch
        if client_id in self.client_prev_model_states:
            pkg["prev_model_state"] = self.client_prev_model_states[
                client_id
            ]
        else:
            pkg["prev_model_state"] = None
        return pkg

    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, dict[str, Any]]
    ):
        # 1) 先走普通 FedAvg 权重聚合
        super().aggregate_client_updates(client_packages)

        # 2) 额外聚合 prototypes
        self._aggregate_prototypes(client_packages)

    # --------------------------- 聚合原型函数 ----------------------------- #
    def _aggregate_prototypes(self, client_packages):
        # ── shape: [num_clients, num_classes, feat_dim] (部分为空 list)
        proto_list = [pkg["prototypes"] for pkg in client_packages.values()]
        weights = torch.tensor(
            [pkg["weight"] for pkg in client_packages.values()],
            dtype=torch.float, device=self.device
        )
        weights /= weights.sum()

        num_classes = len(proto_list[0])
        aggregated = {}
        for c in range(num_classes):
            class_feats = []
            class_wts = []
            for w, plist in zip(weights, proto_list):
                if isinstance(plist[c], torch.Tensor):
                    class_feats.append(plist[c].to(self.device))
                    class_wts.append(w)
            if len(class_feats):
                feats = torch.stack(class_feats, dim=-1)
                class_wts = torch.tensor(class_wts, dtype=torch.float, device=self.device)
                class_wts = class_wts / class_wts.sum()  # 关键
                proto = torch.sum(feats * class_wts, dim=-1)
                aggregated[c] = proto.cpu()
        self.global_prototypes = aggregated