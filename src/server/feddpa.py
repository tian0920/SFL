from omegaconf import DictConfig

from src.client.feddpa import FedDpaClient
from src.server.fedavg import FedAvgServer


class FedDpaServer(FedAvgServer):
    algorithm_name = "FedDpa"
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedDpaClient

    def __init__(self,args: DictConfig):
        super().__init__(args)