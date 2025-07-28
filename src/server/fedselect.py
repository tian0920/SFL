from collections import OrderedDict
import torch
from src.server.fedavg import FedAvgServer
from src.client.fedselect import FedSelectClient


class FedSelectServer(FedAvgServer):
    algorithm_name = "FedSelect"
    client_cls = FedSelectClient

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks = {i: None for i in range(self.client_num)}

    def aggregate_client_updates(self, client_packages):
        agg_params = {}
        count = {}

        for name in self.public_model_params:
            agg_params[name] = torch.zeros_like(self.public_model_params[name])
            count[name] = torch.zeros_like(self.public_model_params[name])

        for client_id, package in client_packages.items():
            model_params = package["regular_model_params"]
            client_mask = package.get("mask")
            self.masks[client_id] = client_mask
            for name in model_params:
                agg_params[name] += model_params[name]
                count[name] += (model_params[name] != 0).float()

        for name in agg_params:
            mask = count[name] > 0
            if mask.any():
                agg_params[name][mask] /= count[name][mask]
                self.public_model_params[name].data[mask] = agg_params[name][mask]

        self.model.load_state_dict(self.public_model_params, strict=False)

    def get_client_model_params(self, client_id):
        mask = self.masks.get(client_id, None)
        regular_params = OrderedDict()
        for k, v in self.public_model_params.items():
            if mask is None or not mask[k].any():
                regular_params[k] = v.clone()
        personal_params = self.clients_personal_model_params[client_id]
        return dict(regular_model_params=regular_params, personal_model_params=personal_params)

    def package(self, client_id):
        pkg = super().package(client_id)
        pkg['mask'] = self.masks[client_id]
        return pkg