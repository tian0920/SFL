from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel


class Embedding(nn.Embedding):
    def __init__(self, num_embedding, embedding_size, embedding_mode, *args, **kwargs):
        super().__init__(num_embedding, embedding_size, *args, **kwargs)
        self.num_iters = 200
        if num_embedding <= embedding_size:
            with torch.no_grad():
                self.weight.data = self.gram_schmidt(self.weight.data)
        else:
            self.weight.data = self.ebv(self.weight.data)
        if embedding_mode == 't':
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def gram_schmidt(self, vectors, eps=1e-10):
        basis = []
        for v in vectors:
            w = v.clone()
            for u in basis:
                w -= torch.dot(u, v) * u
            w_norm = torch.linalg.norm(w)
            w /= max(w_norm, eps)
            basis.append(w)
        basis = torch.stack(basis, dim=0)
        return basis

    def ebv(self, vectors):
        vectors.requires_grad = True
        optimizer = optim.SGD([vectors], lr=1, momentum=0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_iters,
                                                         eta_min=0)
        for i in range(self.num_iters):
            norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
            norm_vectors = vectors / norm
            cosine = norm_vectors @ norm_vectors.t()
            cosine = torch.triu(cosine, diagonal=1)
            row_idx, col_idx = torch.triu_indices(*cosine.size(), offset=1)
            cosine = cosine[row_idx, col_idx]
            loss = cosine.abs().sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        evd = vectors / norm
        return evd


class SFLASClient(FedAvgClient):
    def __init__(self, **commons):
        commons["model"] = FedASModel(
            commons["model"]
        )
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)

    def package(self):
        client_package = super().package()
        client_package["prev_model_state"] = deepcopy(self.model.state_dict())
        return client_package

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
        if not self.testing:
            self.align_federated_parameters(package)
        else:
            self.align_federated_parameters(package)
            # self.model.load_state_dict(self.prev_model.state_dict())

    def align_federated_parameters(self, package):
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        self.dataset.train()

        prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)

                for y, feat in zip(y, features):
                    prototypes[y].append(feat)

        mean_prototypes = [
            torch.stack(prototype).mean(dim=0) if prototype else None
            for prototype in prototypes
        ]

        alignment_optimizer = torch.optim.SGD(
            self.model.base.parameters(), lr=self.args.sflas.alignment_lr
        )

        for _ in range(self.args.sflas.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x, detach=False)
                loss = 0
                for label in y.unique().tolist():
                    if mean_prototypes[label] is not None:
                        loss += F.mse_loss(
                            features[y == label].mean(dim=0), mean_prototypes[label]
                        )

                alignment_optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), max_norm=10.0)  # 需要添加梯度裁剪，必须加，会影响效果注意

                alignment_optimizer.step()

        self.prev_model.cpu()


class FedASModel(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, freeze_generic_classifier=True):
        super().__init__()
        self.base = deepcopy(generic_model.base)
        self.classifier = deepcopy(generic_model.classifier)
        self.freeze_generic_classifier = freeze_generic_classifier

        # 获取特征维度和类别数
        feat_dim = generic_model.classifier.weight.shape[1]     # 输入特征维度
        num_classes = generic_model.classifier.weight.shape[0]  # 类别数

        if self.freeze_generic_classifier:
            embedding_mode = 'f'
        else:
            embedding_mode = 't'

        # 初始化Embedding层 权重矩阵 (num_classes, feat_dim)
        self.target_embedding = Embedding(
            num_embedding=num_classes,              # 类别数量
            embedding_size=feat_dim,                # 特征维度
            embedding_mode=embedding_mode           # 固定embedding权重
        )

    def forward(self, x):
        last_features = self.base(x)                 # 获取特征

        if self.freeze_generic_classifier:
            logit = self.target_embedding.weight @ last_features.t()  # 直接使用embedding权重
            logit = logit.t()
        else:
            logit = self.classifier(last_features)

        return logit