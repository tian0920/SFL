from collections import OrderedDict, deque
from typing import Any, Callable, Optional

import ray
import ray.actor

from src.utils.constants import MODE


class FLbenchTrainer:
    def __init__(
        self, server, client_cls, mode: str, num_workers: int, init_args: dict
    ):
        self.server = server
        self.client_cls = client_cls
        self.mode = mode
        self.current_model = None
        self.num_workers = num_workers
        if self.mode == MODE.SERIAL or self.mode == MODE.SEQUENTIAL or self.mode == MODE.PER_SEQUENTIAL:
            self.worker = client_cls(**init_args)
        elif self.mode == MODE.PARALLEL:
            ray_client = ray.remote(client_cls).options(
                num_cpus=self.server.args.parallel.num_cpus / self.num_workers,
                num_gpus=self.server.args.parallel.num_gpus / self.num_workers,
            )
            self.workers: list[ray.actor.ActorHandle] = [
                ray_client.remote(**init_args) for _ in range(self.num_workers)
            ]
        else:
            raise ValueError(f"Unrecongnized running mode.")

        if self.mode == MODE.SERIAL:
            self.train = self._serial_train
            self.test = self._serial_test
            self.exec = self._serial_exec
        elif self.mode == MODE.SEQUENTIAL:
            self.train = self._sequential_train
            self.test = self._sequential_test
        elif self.mode == MODE.PER_SEQUENTIAL:
            self.train = self._per_sequential_train
            self.test = self._sequential_test
        else:
            self.train = self._parallel_train
            self.test = self._parallel_test
            self.exec = self._parallel_exec

    def _serial_train(self):
        client_packages = OrderedDict()
        for client_id in self.server.selected_clients:
            server_package = self.server.package(client_id)
            client_package = self.worker.train(server_package)
            client_packages[client_id] = client_package

            if self.server.verbose:
                self.server.logger.log(
                    *client_package["eval_results"]["message"], sep="\n"
                )
            self.server.client_metrics[client_id][self.server.current_epoch] = (
                client_package["eval_results"]
            )
            self.server.clients_personal_model_params[client_id].update(
                client_package["personal_model_params"]
            )
            self.server.client_optimizer_states[client_id].update(
                client_package["optimizer_state"]
            )
            self.server.client_lr_scheduler_states[client_id].update(
                client_package["lr_scheduler_state"]
            )

        return client_packages

    def _sequential_train(self):
        """SFL 训练过程：按顺序训练每个客户端，模型参数依次传递"""
        client_packages = OrderedDict()

        for client_id in self.server.selected_clients:
            server_package = self.server.package(client_id)

            # 在 SFL 中，每个客户端接收前一个客户端训练后的模型
            if self.current_model:  # 不是第一个客户端
                server_package["regular_model_params"] = self.current_model  # 传递上一个客户端的模型

            client_package = self.worker.train(server_package)  # 客户端训练
            client_packages[client_id] = client_package

            # 更新最新的模型参数（用于下一个客户端）
            self.current_model = client_package["regular_model_params"]

            if self.server.verbose:
                self.server.logger.log(
                    *client_package["eval_results"]["message"], sep="\n"
                )
            self.server.client_metrics[client_id][self.server.current_epoch] = (
                client_package["eval_results"]
            )
            self.server.clients_personal_model_params[client_id].update(
                client_package["personal_model_params"]
            )
            self.server.client_optimizer_states[client_id].update(
                client_package["optimizer_state"]
            )
            self.server.client_lr_scheduler_states[client_id].update(
                client_package["lr_scheduler_state"]
            )

        last_client_id = self.server.selected_clients[-1]
        next_epoch_parms = client_packages[last_client_id]["regular_model_params"]
        return next_epoch_parms

    def _per_sequential_train(self):
        """SFL 训练过程：按顺序训练每个客户端，模型参数依次传递"""
        client_packages = OrderedDict()

        for client_id in self.server.selected_clients:
            server_package = self.server.package(client_id)

            # 在 SFL 中，每个客户端接收前一个客户端训练后的模型
            if self.current_model:  # 不是第一个客户端
                server_package["regular_model_params"] = self.current_model  # 传递上一个客户端的模型

            client_package = self.worker.train(server_package)  # 客户端训练
            client_packages[client_id] = client_package

            # 更新最新的模型参数（用于下一个客户端）
            self.current_model = client_package["regular_model_params"]

            if self.server.verbose:
                self.server.logger.log(
                    *client_package["eval_results"]["message"], sep="\n"
                )
            self.server.client_metrics[client_id][self.server.current_epoch] = (
                client_package["eval_results"]
            )
            self.server.clients_personal_model_params[client_id].update(
                client_package["personal_model_params"]
            )
            self.server.client_optimizer_states[client_id].update(
                client_package["optimizer_state"]
            )
            self.server.client_lr_scheduler_states[client_id].update(
                client_package["lr_scheduler_state"]
            )
        return client_packages

    def _parallel_train(self):
        clients = self.server.selected_clients
        i = 0
        futures = []
        idle_workers = deque(range(self.num_workers))
        job_map = {}
        client_packages = OrderedDict()
        while i < len(clients) or len(futures) > 0:
            while i < len(clients) and len(idle_workers) > 0:
                worker_id = idle_workers.popleft()
                server_package = ray.put(self.server.package(clients[i]))
                future = self.workers[worker_id].train.remote(server_package)
                job_map[future] = (clients[i], worker_id)
                futures.append(future)
                i += 1

            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    client_id, worker_id = job_map[finished]
                    client_package = ray.get(finished)
                    idle_workers.append(worker_id)
                    client_packages[client_id] = client_package

                    if self.server.verbose:
                        self.server.logger.log(
                            *client_package["eval_results"]["message"], sep="\n"
                        )

                    self.server.client_metrics[client_id][self.server.current_epoch] = (
                        client_package["eval_results"]
                    )
                    self.server.clients_personal_model_params[client_id].update(
                        client_package["personal_model_params"]
                    )
                    self.server.client_optimizer_states[client_id].update(
                        client_package["optimizer_state"]
                    )
                    self.server.client_lr_scheduler_states[client_id].update(
                        client_package["lr_scheduler_state"]
                    )

        return client_packages

    def _serial_test(self, clients: list[int], results: dict):
        for client_id in clients:
            server_package = self.server.package(client_id)
            metrics = self.worker.test(server_package)
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    results[stage][split].update(metrics[stage][split])

    def _sequential_test(self, clients: list[int], results: dict):
        """SFL 测试过程：按顺序测试每个客户端，模型参数依次传递"""
        for client_id in clients:
            server_package = self.server.package(client_id)
            server_package['regular_model_params'] = self.current_model
            metrics = self.worker.test(server_package)
            for stage in ["before", "after"]:
                for split in ["train", "val", "test"]:
                    results[stage][split].update(metrics[stage][split])

    def _parallel_test(self, clients: list[int], results: dict):
        i = 0
        futures = []
        idle_workers = deque(range(self.num_workers))
        job_map = {}  # {future: (client_id, worker_id)}
        while i < len(clients) or len(futures) > 0:
            while i < len(clients) and len(idle_workers) > 0:
                server_package = ray.put(self.server.package(clients[i]))
                worker_id = idle_workers.popleft()
                future = self.workers[worker_id].test.remote(server_package)
                job_map[future] = (clients[i], worker_id)
                futures.append(future)
                i += 1

            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    metrics = ray.get(finished)
                    _, worker_id = job_map[finished]
                    idle_workers.append(worker_id)
                    for stage in ["before", "after"]:
                        for split in ["train", "val", "test"]:
                            results[stage][split].update(metrics[stage][split])

    def _serial_exec(
        self,
        func_name: str,
        clients: list[int],
        package_func: Optional[Callable[[int], dict[str, Any]]] = None,
    ):
        if package_func is None:
            package_func = getattr(self.server, "package")
        client_packages = OrderedDict()
        for client_id in clients:
            server_package = package_func(client_id)
            package = getattr(self.worker, func_name)(server_package)
            client_packages[client_id] = package
        return client_packages

    def _parallel_exec(
        self,
        func_name: str,
        clients: list[int],
        package_func: Optional[Callable[[int], dict[str, Any]]] = None,
    ):
        if package_func is None:
            package_func = getattr(self.server, "package")
        client_packages = OrderedDict()
        i = 0
        futures = []
        idle_workers = deque(range(self.num_workers))
        job_map = {}  # {future: (client_id, worker_id)}
        while i < len(clients) or len(futures) > 0:
            while i < len(clients) and len(idle_workers) > 0:
                server_package = ray.put(package_func(clients[i]))
                worker_id = idle_workers.popleft()
                future = getattr(self.workers[worker_id], func_name).remote(
                    server_package
                )
                job_map[future] = (clients[i], worker_id)
                futures.append(future)
                i += 1

            if len(futures) > 0:
                all_finished, futures = ray.wait(futures)
                for finished in all_finished:
                    package = ray.get(finished)
                    client_id, worker_id = job_map[finished]
                    idle_workers.append(worker_id)
                    client_packages[client_id] = package

        return client_packages
