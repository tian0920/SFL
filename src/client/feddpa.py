from copy import deepcopy
from typing import Any, Dict
import torch
from torch import Tensor
import torch.autograd as autograd

from src.client.fedavg import FedAvgClient


class FedDpaClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)

        # Initialize device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move the model to the specified device

        # Fisher threshold for parameter selection
        self.fisher_threshold = self.args.feddpa.fisher_threshold

        # Initialize client IDs (e.g., 100 clients)
        self.client_ids = list(range(100))  # Client IDs from 0 to 99

        # Store state dictionaries for pre- and post-training
        self.postrain_state_dict = {
            client_id: deepcopy(self.model.state_dict())
            for client_id in self.client_ids
        }

    def set_parameters(self, package: Dict[str, Any]):
        """
        Update the local model parameters based on the server package.
        """
        self.client_id = package["client_id"]
        self.load_data_indices()  # Load data indices for local training

        # Load optimizer state
        if package.get("optimizer_state") and not self.args.common.reset_optimizer_on_global_epoch:
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        # Load learning rate scheduler state if it exists
        if self.lr_scheduler is not None:
            scheduler_state = package.get("lr_scheduler_state", self.init_lr_scheduler_state)
            self.lr_scheduler.load_state_dict(scheduler_state)

        global_regular_params = deepcopy(package.get("regular_model_params", {}))

        # Split parameters into u and v (local model)
        u_loc, v_loc = [], []
        tmp_model = deepcopy(self.model)
        tmp_model.load_state_dict(self.postrain_state_dict[self.client_id])  # Load state dictionary

        fisher_diag = self.compute_fisher_diag(self.trainloader)

        # Split parameters into u and v (local model)
        u_loc, v_loc = [], []
        for param, fisher_value in zip(tmp_model.parameters(), fisher_diag):
            param = param.to(self.device)
            quantile_value = torch.quantile(fisher_value,
                                            self.fisher_threshold)  # Calculate the quantile based on the fisher value
            u_param = (param * (fisher_value > quantile_value)).clone().detach()
            v_param = (param * (fisher_value <= quantile_value)).clone().detach()
            u_loc.append(u_param)
            v_loc.append(v_param)

        # Split parameters into u and v (global model)
        u_glob, v_glob = [], []
        for global_param, fisher_value in zip(global_regular_params.values(), fisher_diag):
            global_param = global_param.to(self.device)
            quantile_value = torch.quantile(fisher_value,
                                            self.fisher_threshold)  # Calculate the quantile for global parameters
            u_param = (global_param * (fisher_value > quantile_value)).clone().detach()
            v_param = (global_param * (fisher_value <= quantile_value)).clone().detach()
            u_glob.append(u_param)
            v_glob.append(v_param)

        # Update local model parameters
        for u_param, v_param, model_param in zip(u_loc, v_glob, tmp_model.parameters()):
            model_param.data = u_param + v_param

        tmp_model = tmp_model.state_dict()
        # Load the updated parameters into the model
        self.model.load_state_dict(tmp_model, strict=True)

    def train(self, server_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the local model using the server package and return the client package.
        """
        self.set_parameters(server_package)

        # Perform local training
        self.train_with_eval()

        # Package and return the results
        return self.package()

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

    def compute_fisher_diag(self, dataloader) -> Dict[str, Tensor]:
        """
        Compute and normalize the Fisher diagonal for the model using a subset of the provided dataloader.

        Args:
        - dataloader (DataLoader): Dataloader containing the dataset.

        Returns:
        - Fisher diagonal (normalized) for the model.
        """
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tmp_model = deepcopy(self.model)
        fisher_diag = [torch.zeros_like(param) for param in tmp_model.parameters()]

        # Limit the loop to a maximum of 10 iterations
        for idx, (data, labels) in enumerate(dataloader):

            data, labels = data.to(device), labels.to(device)

            # Calculate output log probabilities
            log_probs = torch.nn.functional.log_softmax(tmp_model(data), dim=1)

            for i, label in enumerate(labels):
                log_prob = log_probs[i, label]

                # Calculate first-order derivatives (gradients)
                tmp_model.zero_grad()
                grad1 = autograd.grad(log_prob, tmp_model.parameters(), create_graph=True, retain_graph=True)

                # Update Fisher diagonal elements
                for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                    fisher_diag_value.add_(grad_value.detach() ** 2)

                # Free up memory by removing computation graph
                del log_prob, grad1

        # Calculate the mean value
        num_samples = min(len(dataloader), 10)  # Ensure num_samples does not exceed 10
        fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]

        # Normalize Fisher values layer-wise
        normalized_fisher_diag = []
        for fisher_value in fisher_diag:
            x_min = torch.min(fisher_value)
            x_max = torch.max(fisher_value)
            normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
            normalized_fisher_diag.append(normalized_fisher_value)

        return normalized_fisher_diag

    def compute_ig_norm(self, global_regular_params, client_regular_params, device):
        Ig_norm = []  # Store normalized IG values

        for global_param, client_param in zip(global_regular_params.values(), client_regular_params.parameters()):
            client_param = client_param.to(device)
            global_param = global_param.to(device)
            # Compute IG: |(client_param - global_param) * client_param|
            ig_value = torch.abs((client_param - global_param) * client_param)
            # Min-Max normalization
            min_val = torch.min(ig_value)
            max_val = torch.max(ig_value)
            if max_val == min_val:
                # If max == min, return a zero tensor
                normalized_ig = torch.zeros_like(ig_value)
            else:
                normalized_ig = (ig_value - min_val) / (max_val - min_val)

            Ig_norm.append(normalized_ig)  # Add the normalized result to the list

        return Ig_norm