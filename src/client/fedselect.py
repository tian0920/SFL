import torch, math
from src.client.fedavg import FedAvgClient

class FedSelectClient(FedAvgClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = {
            name: torch.zeros_like(param, dtype=torch.bool)
            for name, param in self.model.named_parameters()
        }
        self.alpha = self.args.fedselect.alpha
        self.p = self.args.fedselect.p
        self.last_delta = None
        self.round = 0
        self.update_every = max(1, self.args.fedselect.get("update_every", 5))  # 默认每5轮更新一次掩码

    def _mask_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is None or name not in self.mask:
                continue
            param.grad *= (~self.mask[name]).float()

    def set_parameters(self, package):
        super().set_parameters(package)
        self.round += 1

        if self.last_delta is None:
            return

        # 控制掩码更新频率
        if self.round % self.update_every != 0:
            return

        with torch.no_grad():
            total_params = sum(p.numel() for p in self.mask.values())
            flat_delta = torch.cat([
                v[(~self.mask[k])].flatten()
                for k, v in self.last_delta.items() if k in self.mask and v.numel() > 0
            ])
            if flat_delta.numel() == 0:
                return

            num_select = int(self.p * flat_delta.numel())
            if num_select > 0:
                threshold = torch.topk(flat_delta, num_select).values[-1]
                for k in self.last_delta:
                    if k in self.mask:
                        grow_mask = (self.last_delta[k] >= threshold) & (~self.mask[k])
                        self.mask[k] |= grow_mask

            # alpha 限制裁剪
            personalized_params = sum(m.sum().item() for m in self.mask.values())
            ratio = personalized_params / total_params
            if ratio > self.alpha:
                score_pairs = []
                for k in self.mask:
                    if k in self.last_delta and self.mask[k].any():
                        delta_vals = self.last_delta[k][self.mask[k]].flatten()
                        score_pairs.extend([(k, i, v.item()) for i, v in enumerate(delta_vals)])

                if score_pairs:
                    score_pairs.sort(key=lambda x: -x[2])
                    keep_top = math.ceil(self.alpha * total_params)
                    keep_keys = set((k, i) for k, i, _ in score_pairs[:keep_top])
                    new_mask = {k: torch.zeros_like(v, dtype=torch.bool) for k, v in self.mask.items()}
                    for k, i, _ in score_pairs[:keep_top]:
                        new_mask[k].view(-1)[i] = True
                    self.mask = new_mask

        self.personal_params_name = [k for k, m in self.mask.items() if m.any()]
        self.regular_params_name = [k for k, m in self.mask.items() if not m.any()]

    def fit(self):
        self.model.train()
        self.dataset.train()

        # --- Phase 1: Train personalized parameters ---
        if any(m.any() for m in self.mask.values()):
            for _ in range(self.local_epoch):
                for x, y in self.trainloader:
                    if len(x) <= 1:
                        continue
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for name, param in self.model.named_parameters():
                        if name in self.mask and not self.mask[name].any():
                            param.grad = None
                        elif name in self.mask:
                            param.grad *= self.mask[name].float()
                    self.optimizer.step()

        # --- Phase 2: Train shared parameters ---
        init_state = {
            k: p.clone()
            for k, p in self.model.named_parameters()
            if k in self.mask and (~self.mask[k]).any()
        }

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                for name, param in self.model.named_parameters():
                    if name in self.mask:
                        param.grad *= (~self.mask[name]).float()
                self.optimizer.step()

        if self.lr_scheduler:
            self.lr_scheduler.step()

        # --- Delta update ---
        with torch.no_grad():
            self.last_delta = {
                k: (self.model.state_dict()[k] - init_state[k]).abs()
                for k in init_state
            }

    def package(self):
        pkg = super().package()
        pkg["mask"] = self.mask
        return pkg