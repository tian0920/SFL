from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel


class SFLASClient(FedAvgClient):
    def __init__(self, **commons):
        # commons["model"] = FedASModel(
        #     commons["model"]
        # )
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)

    def package(self):
        client_package = super().package()
        # client_package["weight"] = self.get_fim_trace_sum()
        client_package["weight"] = self.compute_total_param_score()
        client_package["prev_model_state"] = deepcopy(self.model.state_dict())
        return client_package

    def set_parameters(self, package: dict[str, Any]) -> None:
        super().set_parameters(package)
        self.current_epoch = package["current_epoch"]

        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())
        if not self.testing:
            self.align_federated_parameters(package)
        else:
            self.model.load_state_dict(self.prev_model.state_dict())

    # def align_federated_parameters(self, package):   ##### 原版，不加Head蒸馏
    #     self.prev_model.eval()
    #     self.prev_model.to(self.device)
    #     self.model.train()
    #     self.dataset.train()
    #
    #     self.prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
    #
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)
    #
    #             for y, feat in zip(y, features):
    #                 self.prototypes[y].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(prototype).mean(dim=0) if prototype else None
    #         for prototype in self.prototypes
    #     ]
    #
    #     alignment_optimizer = torch.optim.SGD(
    #         self.model.base.parameters(),
    #         lr=self.args.sflas.alignment_lr
    #     )
    #
    #     for _ in range(self.args.sflas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.model.get_last_features(x, detach=False)
    #             loss = 0
    #             for label in y.unique().tolist():
    #                 if mean_prototypes[label] is not None:
    #                     loss += F.mse_loss(
    #                         features[y == label].mean(dim=0), mean_prototypes[label]
    #                     )
    #
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             if self.args.dataset.name == 'cifar100':
    #                 torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), max_norm=10.0)  # 需要添加梯度裁剪，必须加，会影响效果注意
    #             alignment_optimizer.step()
    #
    #     self.prev_model.cpu()

    # def align_federated_parameters(self, package): #### 单教师蒸馏
    #     self.prev_model.eval()  # 将教师模型设置为评估模式
    #     self.prev_model.to(self.device)
    #     self.model.train()  # 将学生模型设置为训练模式
    #     self.model.to(self.device)
    #     self.dataset.train()
    #
    #     self.prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
    #
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)  # 获取教师模型特征
    #             for label, feat in zip(y, features):
    #                 self.prototypes[label].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(prototype).mean(dim=0) if prototype else None
    #         for prototype in self.prototypes
    #     ]
    #
    #     alignment_optimizer = torch.optim.SGD(
    #         list(self.model.base.parameters()) + list(self.model.classifier.parameters()),  # 更新body和classifier部分
    #         lr=self.args.sflas.alignment_lr
    #     )
    #
    #     for _ in range(self.args.sflas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.model.get_last_features(x, detach=False)  # 获取学生模型的特征
    #             loss = 0
    #             for label in y.unique().tolist():
    #                 if mean_prototypes[label] is not None:
    #                     loss += F.mse_loss(
    #                         features[y == label].mean(dim=0), mean_prototypes[label]
    #                     )
    #
    #             # 知识蒸馏部分（只蒸馏head部分）
    #             with torch.no_grad():
    #                 teacher_features = self.prev_model.get_last_features(x, detach=True)  # 获取教师模型的特征
    #                 teacher_output = self.prev_model.classifier(teacher_features)  # 获取教师分类器的输出
    #             student_output = self.model.classifier(features)  # 获取学生分类器的输出
    #
    #             if self.args.dataset.name == "cifar100":
    #                 distillation_loss = F.kl_div(
    #                     F.log_softmax(student_output / 1, dim=1),
    #                     F.softmax(teacher_output / 1, dim=1),
    #                     reduction='batchmean'
    #                 ) * (1 ** 2)
    #             else:
    #                 distillation_loss = F.kl_div(
    #                     F.log_softmax(student_output / self.args.sflas.temperature, dim=1),
    #                     F.softmax(teacher_output / self.args.sflas.temperature, dim=1),
    #                     reduction='batchmean'
    #                 ) * (self.args.sflas.temperature ** 2)
    #
    #             loss += distillation_loss
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #
    #             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    #             # self.adaptive_clip_gradients(self.model)
    #             alignment_optimizer.step()
    #     self.prev_model.cpu()

    def align_federated_parameters(self, package): #### 双教师蒸馏
        # 设置模型状态
        self.prev_model.eval()  # 教师 1
        self.prev_model.to(self.device)

        self.global_model = deepcopy(self.model)  # 教师 2：初始与 self.model 相同
        self.global_model.eval()
        self.global_model.to(self.device)

        self.model.train()  # 学生模型
        self.model.to(self.device)
        self.dataset.train()

        # ------- 1. 构建原型 -------
        self.prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]

        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.prev_model.get_last_features(x)
                for label, feat in zip(y, features):
                    self.prototypes[label].append(feat)

        mean_prototypes = [
            torch.stack(p).mean(dim=0) if p else None
            for p in self.prototypes
        ]

        # ------- 2. 初始化优化器 -------
        alignment_optimizer = torch.optim.SGD(
            list(self.model.base.parameters()) + list(self.model.classifier.parameters()),
            lr=self.args.sflas.alignment_lr
        )

        # ------- 3. 联合优化 -------
        for _ in range(self.args.sflas.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                features = self.model.get_last_features(x, detach=False)

                # 原型对齐 loss
                proto_loss = 0
                for label in y.unique().tolist():
                    if mean_prototypes[label] is not None:
                        proto_loss += F.mse_loss(
                            features[y == label].mean(dim=0), mean_prototypes[label]
                        )

                # 教师 logits（global model + prev model）
                with torch.no_grad():
                    teacher_feat_global = self.global_model.get_last_features(x, detach=True)
                    teacher_logits_global = self.global_model.classifier(teacher_feat_global)

                    teacher_feat_prev = self.prev_model.get_last_features(x, detach=True)
                    teacher_logits_prev = self.prev_model.classifier(teacher_feat_prev)

                student_logits = self.model.classifier(features)

                T = self.args.sflas.temperature
                lambda_g = self.args.sflas.lambda_global  # 权重：global teacher

                kd_loss_global = F.kl_div(
                    F.log_softmax(student_logits / T + 1e-10, dim=1),
                    F.softmax(teacher_logits_global / T + 1e-10, dim=1),
                    reduction='batchmean'
                )

                kd_loss_prev = F.kl_div(
                    F.log_softmax(student_logits / T + 1e-10, dim=1),
                    F.softmax(teacher_logits_prev / T + 1e-10, dim=1),
                    reduction='batchmean'
                )

                distill_loss = (lambda_g * kd_loss_global + (1 - lambda_g) * kd_loss_prev) * (T ** 2)

                # 总 loss
                loss = proto_loss + distill_loss

                alignment_optimizer.zero_grad()
                loss.backward()
                if self.args.dataset.name == "cifar10" or self.args.dataset.name == "cifar100" or self.args.dataset.name == "svhn":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                alignment_optimizer.step()

        self.prev_model.cpu()
        self.global_model.cpu()

    def get_fim_trace_sum(self) -> float:
        self.model.eval()
        self.dataset.eval()

        fim_trace_sum = 0

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = (
                -F.log_softmax(logits, dim=1).gather(dim=1, index=y.unsqueeze(1)).mean()
            )

            self.model.zero_grad()
            loss.backward()

            for param in self.model.parameters():
                if param.grad is not None:
                    fim_trace_sum += (param.grad.data**2).sum().item()

        return fim_trace_sum

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

    def fit(self):
        self.model.train()
        self.dataset.train()

        self.lambda_contrastive = 0.5
        self.lambda_ortho = 5

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                features = self.model.get_last_features(x, detach=False)
                cls_loss = self.criterion(logit, y)
                contrastive_loss = self.compute_proto_contrastive_loss(features, y, temperature=0.2)
                # print(cls_loss, '\n', contrastive_loss, '\n')

                total_loss = cls_loss + self.lambda_contrastive * contrastive_loss

                if hasattr(self.model, "target_embedding"):
                    embedding = self.model.target_embedding.weight  # [num_classes, feat_dim]
                    normed = F.normalize(embedding, p=2, dim=1, eps=1e-8)  # [num_classes, feat_dim]
                    cosine = normed @ normed.T  # [num_classes, num_classes]

                    # 创建一个掩码，去除对角线元素
                    identity = torch.eye(cosine.size(0), device=cosine.device)  # 单位矩阵
                    mask = 1 - identity  # 非对角线元素为1，对角线为0

                    # 只计算非对角线部分的损失
                    off_diag = (cosine - identity) * mask  # 计算非对角线部分的差值
                    orth_loss = self.lambda_ortho * off_diag.pow(2).sum()  # 计算正交性损失

                    # 将正交性损失加入总损失
                    total_loss += orth_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def compute_contrastive_loss(self, features, labels, temperature=0.5):
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)  # [B, B]

        batch_size = features.size(0)
        mask = torch.eye(batch_size, device=features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(features.device)

        similarity_matrix /= temperature

        # Contrastive loss calculation
        logits = similarity_matrix
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss


    def compute_proto_contrastive_loss(self, features, labels, temperature=0.5):
        features = F.normalize(features, dim=1)  # [B, D]
        prototypes = [
            torch.stack(prototype).mean(dim=0) if prototype else torch.zeros(features.size(1), device=self.device)
            for prototype in self.prototypes
        ]
        prototypes = F.normalize(torch.stack(prototypes), dim=1)
        logits = features @ prototypes.T  # 相似度
        logits /= temperature

        loss = F.cross_entropy(logits, labels)
        return loss


class FedASModel(DecoupledModel):
    def __init__(self, generic_model: DecoupledModel, freeze_generic_classifier=False):
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

    def gram_schmidt(self, vectors, eps=1e-8):
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