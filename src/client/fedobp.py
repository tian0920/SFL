from copy import deepcopy
from typing import Any, Dict

import json
import torch
from torch import Tensor
import os

from src.client.fedavg import FedAvgClient


class FedOBPClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # Initialize device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the specified device

        # Collect parameter names and shapes for reconstruction
        state_dict = self.model.state_dict()
        self.all_param_names = list(state_dict.keys())

        # Initialize client IDs (e.g., 100 clients)
        self.client_ids = list(range(100))  # Client IDs from 0 to 99

        self.postrain_state_dict = {
            client_id: deepcopy(self.model.state_dict())
            for client_id in self.client_ids
        }

        self.alpha = self.args.fedobp.alpha

        self.Ig_ema = {}
        self.global_ema_params = {}

        if self.args.fedobp.CLS:
            self.param_names = [name for name in self.all_param_names if "classifier" in name]
        else:
            self.param_names = self.all_param_names

    def train(self, server_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the local model using the server package and return the client package.
        """
        self.set_parameters(server_package)

        # Perform local training
        self.train_with_eval()

        # Package and return the results
        return self.package()

    def set_parameters(self, package: Dict[str, Any]):
        """
        Update the local model parameters based on the server package.
        """
        self.client_id = package["client_id"]
        self.current_epoch = package["current_epoch"]
        self.load_data_indices()  # Load data indices for local training

        ##### 1. Directly set Ig_ratio
        self.ig_ratio = self.args.fedobp.ig_ratio

        # Load optimizer state
        if package.get("optimizer_state") and not self.args.common.reset_optimizer_on_global_epoch:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        # Load learning rate scheduler state if it exists
        if self.lr_scheduler is not None:
            scheduler_state = package.get("lr_scheduler_state", self.init_lr_scheduler_state)
            self.lr_scheduler.load_state_dict(scheduler_state)

        # Retrieve server parameters and compute IG (Importance Gradient)
        global_regular_params = deepcopy(package.get("regular_model_params"))
        client_regular_params = self.postrain_state_dict[self.client_id]

        Ig_norm = {}
        for name, global_param in global_regular_params.items():
            if name not in self.param_names:
                continue
            client_param = client_regular_params[name].to(self.device)
            global_param = global_param.to(self.device)
            current_ig = torch.abs((client_param - global_param) * client_param)

            if self.client_id not in self.Ig_ema:
                self.Ig_ema[self.client_id] = {}  # Initialize the client's dictionary

            # Check if using EMA method
            if self.args.fedobp.EMA:
                if name not in self.Ig_ema[self.client_id]:
                    self.Ig_ema[self.client_id][name] = current_ig.clone()  # Initialize EMA value
                else:
                    self.Ig_ema[self.client_id][name] = (1 - self.alpha) * current_ig + self.alpha * self.Ig_ema[self.client_id][name]
            else:
                self.Ig_ema[self.client_id][name] = current_ig.clone()

            # 1. Check if LayerNorm
            if self.args.fedobp.norm == 'layer':
                Ig_norm[name] = ((self.Ig_ema[self.client_id][name] - torch.min(self.Ig_ema[self.client_id][name]))
                                 / (torch.max(self.Ig_ema[self.client_id][name]) - torch.min(self.Ig_ema[self.client_id][name])))
            else:
                Ig_norm[name] = self.Ig_ema[self.client_id][name]

        # 2. Check if GlobalNorm
        if self.args.fedobp.norm == 'global':
            current_ig_ema = self._concat_parameters(self.Ig_ema[self.client_id])
            for name, value in self.Ig_ema[self.client_id].items():
                Ig_norm[name] = (value - current_ig_ema.min()) / (current_ig_ema.max() - current_ig_ema.min())

        # Calculate quantile Threshold
        Ig_threshold = torch.quantile(self._concat_parameters(Ig_norm), self.ig_ratio)

        new_state_dict = self._update_parameters(
            new_regular_params=global_regular_params,
            In_dict=Ig_norm,      #### layer / global norm
            In_threshold=Ig_threshold,
            old_regular_params=client_regular_params
        )

        # Load the updated parameters into the model
        self.model.load_state_dict(new_state_dict, strict=True)

    def package(self) -> Dict[str, Any]:
        """
        Package the updated model parameters and additional metrics to send back to the server.
        """
        regular_params = {key: param.cpu().clone() for key, param in self.model.state_dict().items()}

        self.postrain_state_dict[self.client_id] = deepcopy(self.model.state_dict())

        return {
            "weight": len(self.trainset),
            "eval_results": self.eval_results,
            "regular_model_params": regular_params,  # Keep consistent with prior files
            "personal_model_params": {},  # Placeholder for personal parameters if any
            "optimizer_state": deepcopy(self.optimizer.state_dict()),
            "lr_scheduler_state": deepcopy(self.lr_scheduler.state_dict()) if self.lr_scheduler else {},
        }

    def _concat_parameters(self, In_dict: Dict[str, Tensor]) -> Tensor:
        """
        Concatenate all tensors in In_dict into a single tensor.
        """
        # Collect all tensors corresponding to param_names
        In_list = [In_dict[name].view(-1) for name in self.all_param_names if name in In_dict]
        if not In_list:
            return torch.tensor([], device=self.device)
        # Concatenate all tensors into one
        In_concat = torch.cat(In_list)
        return In_concat

    def _update_parameters(
            self,
            new_regular_params: Dict[str, Tensor],
            In_dict: Dict[str, Tensor],
            In_threshold: float,
            old_regular_params: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Update the model parameters based on the computed thresholds.
        Additionally, log the mask statistics (True/False counts for each layer's bias and weight).
        The statistics are appended to the existing 'mask_statistics.json' file.
        """
        new_dict = {}
        if self.args.fedobp.track:
            mask_statistics = {
                "conv1": {"weight": {"True": 0, "False": 0}, "bias": {"True": 0, "False": 0}},
                "conv2": {"weight": {"True": 0, "False": 0}, "bias": {"True": 0, "False": 0}},
                "fc1": {"weight": {"True": 0, "False": 0}, "bias": {"True": 0, "False": 0}},
                "classifier": {"weight": {"True": 0, "False": 0}, "bias": {"True": 0, "False": 0}},
            }

        for param_name in self.all_param_names:
            old_param = old_regular_params[param_name].to(self.device)
            new_param = new_regular_params[param_name].to(self.device)

            if param_name not in self.param_names:
                new_dict[param_name] = new_param
                continue

            In_tensor = In_dict.get(param_name)

            # Create a mask where the importance metric is below the threshold
            mask = In_tensor <= In_threshold
            updated_param = torch.where(mask, new_param, old_param)
            new_dict[param_name] = updated_param

            if self.args.fedobp.track:
                # Track statistics for each layer's weight and bias
                if "weight" in param_name:
                    # Extract the layer name properly (handling classifier separately)
                    if "classifier" in param_name:
                        layer_name = "classifier"
                    else:
                        layer_name = param_name.split('.')[1]  # Extract layer name (e.g., 'conv1')

                    true_count = mask.sum().item()
                    false_count = mask.numel() - true_count
                    mask_statistics[layer_name]["weight"]["True"] += true_count
                    mask_statistics[layer_name]["weight"]["False"] += false_count
                elif "bias" in param_name:
                    # Extract the layer name properly (handling classifier separately)
                    if "classifier" in param_name:
                        layer_name = "classifier"
                    else:
                        layer_name = param_name.split('.')[1]  # Extract layer name (e.g., 'conv1')

                    true_count = mask.sum().item()
                    false_count = mask.numel() - true_count
                    mask_statistics[layer_name]["bias"]["True"] += true_count
                    mask_statistics[layer_name]["bias"]["False"] += false_count

        if self.args.fedobp.track:
            # Check if the file already exists, and if not, create it with an empty list.
            file_exists = os.path.isfile(f"{self.args.dataset.name}_mask_statistics.json")
            if file_exists:
                with open(f"{self.args.dataset.name}_mask_statistics.json", "r") as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            # Append the new statistics to the existing data.
            existing_data.append(mask_statistics)

            # Write the updated statistics back to the file
            with open(f"{self.args.dataset.name}_mask_statistics.json", "w") as f:
                json.dump(existing_data, f, indent=4)

        # Print out the statistics for verification
        # print("Mask Statistics (True/False counts per layer):")
        # for layer, stats in mask_statistics.items():
        #     print(f"{layer}:")
        #     for param_type, counts in stats.items():
        #         print(f"  {param_type} -> True: {counts['True']}, False: {counts['False']}")

        return new_dict