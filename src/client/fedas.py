from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES
from src.utils.models import DecoupledModel


class FedASClient(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.prev_model: DecoupledModel = deepcopy(self.model)

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

    def package(self):
        client_package = super().package()
        # FedAS uses the sum of FIM traces as the weight
        # client_package["weight"] = self.get_fim_trace_sum()
        # client_package["weight"] = self.compute_total_param_score()
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
            # FedAS evaluates clients' personalized models
            self.model.load_state_dict(self.prev_model.state_dict())

    # def align_federated_parameters(self):  ###### 原fedas代码
    #     self.prev_model.eval()
    #     self.prev_model.to(self.device)
    #     self.model.train()
    #     self.dataset.train()
    #
    #     prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
    #
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)
    #
    #             for y, feat in zip(y, features):
    #                 prototypes[y].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(prototype).mean(dim=0) if prototype else None
    #         for prototype in prototypes
    #     ]
    #
    #     alignment_optimizer = torch.optim.SGD(
    #         self.model.base.parameters(), lr=self.args.fedas.alignment_lr
    #     )
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
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
    #             alignment_optimizer.step()
    #
    #     self.prev_model.cpu()

    # def align_federated_parameters(self):  # #### 原型增强 + 两阶段优化 + head蒸馏
    #     self.prev_model.eval()
    #     self.prev_model.to(self.device)
    #     self.model.train()
    #     self.model.to(self.device)
    #     self.dataset.train()
    #
    #     self.prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
    #
    #     # 构建教师模型原型
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)
    #             for label, feat in zip(y, features):
    #                 self.prototypes[label].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(prototype).mean(dim=0) if prototype else torch.zeros_like(features[0])
    #         for prototype in self.prototypes
    #     ]
    #
    #     ###### 第一阶段：仅训练 base，优化原型对齐损失 ######
    #     alignment_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.model.get_last_features(x, detach=False)
    #
    #             loss = 0
    #             for label in y.unique().tolist():
    #                 if mean_prototypes[label] is not None:
    #                     loss += F.mse_loss(
    #                         features[y == label].mean(dim=0), mean_prototypes[label]
    #                     )
    #
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             # if self.args.dataset.name in ["cifar10", "cifar100", "svhn"]:
    #             torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), max_norm=10.0)
    #             alignment_optimizer.step()
    #
    #     ###### 第二阶段：冻结 base，仅训练 classifier，通过增强原型蒸馏 classifier ######
    #     self.model.base.eval()  # 冻结base
    #
    #     distill_optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #
    #             with torch.no_grad():
    #                 raw_features = self.model.get_last_features(x)  # base输出
    #                 enhanced_features = raw_features + torch.stack([mean_prototypes[label] for label in y])  # 构造增强特征
    #
    #                 teacher_features = self.prev_model.get_last_features(x, detach=True)
    #                 teacher_output = self.prev_model.classifier(teacher_features)
    #
    #             # 学生head使用增强特征
    #             student_output = self.model.classifier(enhanced_features)
    #
    #             distillation_loss = F.kl_div(
    #                 F.log_softmax(student_output / self.args.fedas.distill_temperature, dim=1),
    #                 F.softmax(teacher_output / self.args.fedas.distill_temperature, dim=1),
    #                 reduction='batchmean'
    #             ) * (self.args.fedas.distill_temperature ** 2)
    #
    #             distill_optimizer.zero_grad()
    #             distillation_loss.backward()
    #             # if self.args.dataset.name in ["cifar10", "cifar100", "svhn"]:
    #             torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=10.0)
    #             distill_optimizer.step()
    #
    #     self.prev_model.cpu()

    # def align_federated_parameters(self):  # #### 原型增强 + 两阶段优化 + head微调
    #     self.prev_model.eval()
    #     self.prev_model.to(self.device)
    #     self.model.train()
    #     self.model.to(self.device)
    #     self.dataset.train()
    #
    #     self.prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
    #
    #     # 构建教师模型原型
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)
    #             for label, feat in zip(y, features):
    #                 self.prototypes[label].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(prototype).mean(dim=0) if prototype else torch.zeros_like(features[0])
    #         for prototype in self.prototypes
    #     ]
    #
    #     ###### 第一阶段：仅训练 base，优化原型对齐损失 ######
    #     alignment_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.model.get_last_features(x, detach=False)
    #
    #             loss = 0
    #             for label in y.unique().tolist():
    #                 if mean_prototypes[label] is not None:
    #                     loss += F.mse_loss(
    #                         features[y == label].mean(dim=0), mean_prototypes[label]
    #                     )
    #
    #             alignment_optimizer.zero_grad()
    #             loss.backward()
    #             if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
    #                 torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), max_norm=10.0)
    #             alignment_optimizer.step()
    #
    #     ###### 第二阶段：冻结 base，仅训练 classifier，通过增强特征微调分类器 ######
    #     self.model.base.eval()  # 冻结 base
    #
    #     classifier_optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #
    #             # 计算原始特征
    #             raw_features = self.model.get_last_features(x)
    #
    #             # 构造增强特征（new特征 + 原型特征）
    #             enhanced_features = raw_features + torch.stack([mean_prototypes[label] for label in y])
    #
    #             # 计算分类损失
    #             outputs = self.model.classifier(enhanced_features)
    #             classification_loss = F.cross_entropy(outputs, y)
    #
    #             classifier_optimizer.zero_grad()
    #             classification_loss.backward()
    #             if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
    #                 torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), max_norm=10.0)
    #             classifier_optimizer.step()
    #
    #     self.prev_model.cpu()

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

        mean_prototypes = [
            torch.stack(p).mean(0) if p else torch.zeros_like(features[0])
            for p in self.prototypes
        ]
        mean_global_prototypes = [
            torch.stack(p).mean(0) if p else torch.zeros_like(global_features[0])
            for p in self.global_prototypes
        ]

        #### 第一阶段：对齐 base 层（不变）
        base_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.args.fedas.alignment_lr)
        for _ in range(self.args.fedas.alignment_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                feats = self.model.get_last_features(x, detach=False)

                loss = sum(
                    F.mse_loss(feats[y == label].mean(0), mean_prototypes[label])
                    for label in y.unique().tolist() if mean_prototypes[label] is not None
                )

                base_optimizer.zero_grad()
                loss.backward()
                if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
                    torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), 10.0)
                base_optimizer.step()

        #### 第二阶段：联合训练 base + classifier + generator
        self.model.train()  # 再次确保全模型为训练态

        generator = LinearGenerator().to(self.device)
        generator.train()

        all_params = list(self.model.base.parameters()) + \
                     list(self.model.classifier.parameters()) + \
                     list(generator.parameters())

        joint_optimizer = torch.optim.SGD(all_params, lr=self.args.fedas.alignment_lr)

        for _ in range(self.args.fedas.headfinetune_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    t1 = self.global_model.get_last_features(x)
                    t2 = torch.stack([mean_prototypes[label] for label in y])
                    t3 = self.prev_model.get_last_features(x)

                s1 = self.model.get_last_features(x)
                s2 = torch.stack([mean_global_prototypes[label] for label in y])

                combined = generator(t1, t2, s1, s2, t3)
                output = self.model.classifier(combined)

                loss = F.cross_entropy(output, y)

                joint_optimizer.zero_grad()
                loss.backward()
                if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
                    torch.nn.utils.clip_grad_norm_(all_params, 10.0)
                joint_optimizer.step()

        self.prev_model.cpu()

    # def align_federated_parameters(self):
    #     self.prev_model.eval().to(self.device)
    #     self.model.train().to(self.device)
    #     self.dataset.train()
    #     self.global_model = deepcopy(self.model)
    #     self.global_model.eval()
    #
    #     # 构建原型向量
    #     num_classes = NUM_CLASSES[self.args.dataset.name]
    #     self.prototypes = [[] for _ in range(num_classes)]
    #     self.global_prototypes = [[] for _ in range(num_classes)]
    #
    #     with torch.no_grad():
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             features = self.prev_model.get_last_features(x)
    #             global_features = self.model.get_last_features(x)
    #             for label, feat in zip(y, features):
    #                 self.prototypes[label].append(feat)
    #             for label, feat in zip(y, global_features):
    #                 self.global_prototypes[label].append(feat)
    #
    #     mean_prototypes = [
    #         torch.stack(p).mean(0) if p else torch.zeros_like(features[0])
    #         for p in self.prototypes
    #     ]
    #
    #     mean_global_prototypes = [
    #         torch.stack(p).mean(0) if p else torch.zeros_like(global_features[0])
    #         for p in self.global_prototypes
    #     ]
    #
    #     ### 第一阶段：对齐 base 层
    #     base_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     for _ in range(self.args.fedas.alignment_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #             feats = self.model.get_last_features(x, detach=False)
    #
    #             loss = sum(
    #                 F.mse_loss(feats[y == label].mean(0), mean_prototypes[label])
    #                 for label in y.unique().tolist() if mean_prototypes[label] is not None
    #             )
    #
    #             base_optimizer.zero_grad()
    #             loss.backward()
    #             if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
    #                 torch.nn.utils.clip_grad_norm_(self.model.base.parameters(), 10.0)
    #             base_optimizer.step()
    #
    #     ### 第二阶段：冻结 base，仅微调 head，固定生成器
    #     self.model.base.eval()
    #
    #     # 初始化并冻结生成器
    #     # sample_x = next(iter(self.trainloader))[0].to(self.device)
    #     # with torch.no_grad():
    #     #     dim = self.model.get_last_features(sample_x).shape[1]
    #     # generator = LinearGenerator(dim).to(self.device)   #### 线性生成器
    #
    #     generator = LinearGenerator().to(self.device)   #### 线性生成器
    #
    #     # generator.eval()
    #     # for p in generator.parameters():
    #     #     p.requires_grad = False
    #
    #     generator.train()
    #     generator_optimizer = torch.optim.SGD(generator.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     classifier_optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.args.fedas.alignment_lr)
    #
    #     # self.model.classifier.eval()
    #     # for p in self.model.classifier.parameters():
    #     #     p.requires_grad = False
    #
    #     for _ in range(self.args.fedas.headfinetune_epoch):
    #         for x, y in self.trainloader:
    #             x, y = x.to(self.device), y.to(self.device)
    #
    #             with torch.no_grad():
    #                 t1 = self.global_model.get_last_features(x)
    #                 t2 = torch.stack([mean_prototypes[label] for label in y])
    #                 s1 = self.model.get_last_features(x)
    #                 s2 = torch.stack([mean_global_prototypes[label] for label in y])
    #                 t3 = self.prev_model.get_last_features(x)
    #                 combined = generator(t1, t2, s1, s2, t3)
    #
    #             output = self.model.classifier(combined)
    #             loss = F.cross_entropy(output, y)
    #
    #             generator_optimizer.zero_grad()
    #             classifier_optimizer.zero_grad()
    #
    #             loss.backward()
    #             if self.args.dataset.name in ["cifar10", "cifar100", "cinic10"]:
    #                 torch.nn.utils.clip_grad_norm_(self.model.classifier.parameters(), 10.0)
    #             generator_optimizer.step()
    #             classifier_optimizer.step()
    #
    #     self.prev_model.cpu()

    def fit(self):
        self.model.train()
        self.dataset.train()
        self.lambda_contrastive = self.args.fedas.lambda_contrastive

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                features = self.model.get_last_features(x, detach=False)
                cls_loss = self.criterion(logit, y)
                contrastive_loss = self.compute_proto_contrastive_loss(features, y, temperature=self.args.fedas.com_temperature)

                total_loss = cls_loss + self.lambda_contrastive * contrastive_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

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

class LinearGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(5))  # 可训练权重

    def forward(self, t1, t2, s1, s2, t3):
        w = F.softmax(self.weights, dim=0)  # 保证和为1
        return w[0]*t1 + w[1]*t2 + w[2]*s1 + w[3]*s2 + w[4]*t3