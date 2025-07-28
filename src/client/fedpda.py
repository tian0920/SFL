from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel


class FedPDAClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)

    def compute_total_param_score(self):
        device = next(self.model.parameters()).device
        self.prev_model.to(device)  # 👈 把 prev_model 移到 model 所在设备

        total = 0.0
        for (param, prev_param) in zip(self.model.parameters(), self.prev_model.parameters()):
            if param.requires_grad:
                diff = param.data - prev_param.data
                # product = diff * param.data
                total += diff.abs().sum().item()
        return total

    def package(self):
        client_package = super().package()
        # fedpda uses the sum of FIM traces as the weight
        client_package["weight"] = self.compute_total_param_score()
        client_package["prev_model_state"] = deepcopy(self.model.state_dict())
        return client_package

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
        if not self.testing:
            self.align_federated_parameters()
        else:
            # fedpda evaluates clients' personalized models
            self.model.load_state_dict(self.prev_model.state_dict())

    def align_federated_parameters(self):
        self.prev_model.eval().to(self.device)
        self.model.train().to(self.device)
        self.dataset.train()

        self.global_model = deepcopy(self.model)
        self.global_model.eval()

        num_classes = NUM_CLASSES[self.args.dataset.name]
        self.prototypes = [[] for _ in range(num_classes)]
        self.global_prototypes = [[] for _ in range(num_classes)]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)
                global_features = self.model.get_last_features(x)
                for label, feat in zip(y, features):
                    self.prototypes[label].append(feat)
                for label, feat in zip(y, global_features):
                    self.global_prototypes[label].append(feat)

        self.mean_prototypes = [
            torch.stack(p).mean(0) if p else torch.zeros_like(features[0])
            for p in self.prototypes
        ]
        self.mean_global_prototypes = [
            torch.stack(p).mean(0) if p else torch.zeros_like(global_features[0])
            for p in self.global_prototypes
        ]

        ### 第一阶段：对齐 base 层（不变）
        base_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.args.fedpda.alignment_lr)
        for _ in range(self.args.fedpda.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                feats = self.model.get_last_features(x, detach=False)
                FEATURE_DIM = feats[0].shape[0]
                loss = sum(
                    F.mse_loss(feats[y == label].mean(0), self.mean_prototypes[label])
                    for label in y.unique().tolist() if self.mean_prototypes[label] is not None
                )
                contrastive_loss = self.compute_proto_contrastive_loss(feats, y, temperature=self.args.fedpda.com_temperature)
                total_loss = loss + self.args.fedpda.lambda_contrastive * contrastive_loss

                base_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), 10.0)
                base_optimizer.step()

        ### 第二阶段：联合训练 base + classifier + generator
        self.model.train()  # 再次确保全模型为训练态

        # generator = LinearGenerator().to(self.device)
        generator = NonlinearGenerator(feature_dim=FEATURE_DIM).to(self.device)

        generator.train()

        all_params = list(self.model.base.parameters()) + \
                     list(self.model.classifier.parameters()) + \
                     list(generator.parameters())

        joint_optimizer = torch.optim.SGD(all_params, lr=self.args.fedpda.alignment_lr)

        for _ in range(self.args.fedpda.headfinetune_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    t1 = self.global_model.get_last_features(x)
                    t2 = torch.stack([self.mean_prototypes[label] for label in y])

                s1 = self.model.get_last_features(x)
                s2 = torch.stack([self.mean_global_prototypes[label] for label in y])

                combined = generator(t1, t2, s1, s2)
                output = self.model.classifier(combined)

                loss = F.cross_entropy(output, y)

                joint_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 10.0)
                joint_optimizer.step()

        self.prev_model.cpu()

    def compute_proto_contrastive_loss(self, features, labels, temperature=0.5):
        features = F.normalize(features, dim=1)  # [B, D]
        prototypes = F.normalize(torch.stack(self.mean_prototypes), dim=1)
        logits = features @ prototypes.T  # 相似度
        logits /= temperature

        loss = F.cross_entropy(logits, labels)
        return loss

class LinearGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(4))  # 可训练权重

    def forward(self, t1, t2, s1, s2):
        w = F.softmax(self.weights, dim=0)  # 保证和为1
        return w[0]*t1 + w[1]*t2 + w[2]*s1 + w[3]*s2

class NonlinearGenerator(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, t1, t2, s1, s2):
        x = torch.cat([t1, t2, s1, s2], dim=1)  # [B, 4D]
        return self.mlp(x)