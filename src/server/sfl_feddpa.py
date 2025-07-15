from src.client.feddpa import FedDpaClient
from omegaconf import DictConfig
from copy import deepcopy
from src.server.fedavg import FedAvgServer
from src.utils.trainer import FLbenchTrainer
from src.utils.constants import MODE


class FedDpaServer(FedAvgServer):
    algorithm_name = "FedDpa"
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedDpaClient

    def __init__(self, args: DictConfig):
        args.mode = "sequential"
        super().__init__(args)

    def init_trainer(self, **extras):
        """Initiate the FL-bench trainier that responsible to client training.

        Args:
            `extras`: Arguments of `self.client_cls.__init__()` that NOT included in
        `[model, args, optimizer_cls, lr_scheduler_cls, dataset, data_indices,
        device, return_diff]`.
        """
        if self.args.mode == "sequential":
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.SEQUENTIAL,
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
        last_item = models.popitem(last=True)  # Returns ('a', 1) and removes it
        self.model.load_state_dict(last_item[1]['regular_model_params'], strict=False)