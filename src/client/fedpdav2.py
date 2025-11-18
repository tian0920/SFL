from __future__ import annotations
from typing import Any, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.client.fedavg import FedAvgClient
from src.utils.constants import NUM_CLASSES

import matplotlib.font_manager as fm
font_path = "C:/Windows/Fonts/times.ttf"
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()


class LinearGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4))  # 权重参数，用于加权每个输入

    def forward(self, feats, prev_feats, proto_g, proto_c):
        # softmax 用于对每个输入的权重进行归一化
        w = F.softmax(self.w, dim=0)
        # 返回加权和的结果
        return w[0] * feats + w[1] * prev_feats + w[2] * proto_g + w[3] * proto_c


class NonlinearGenerator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 1024):
        super(NonlinearGenerator, self).__init__()
        # 定义一个多层感知机（MLP），用于处理特征和原型的组合
        self.fc1 = nn.Linear(4 * feature_dim, hidden_dim)  # 输入 4 * feature_dim，输出 hidden_dim
        self.fc2 = nn.Linear(hidden_dim, feature_dim)  # 输出 feature_dim
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, feats, prev_feats, proto_g, proto_c):
        # 扩展 proto_g 和 proto_c 为与 feats 和 prev_feats 相同的形状
        proto_g_exp = proto_g.expand(feats.size(0), feats.size(1))  # 扩展为 (x, feature_dim)
        proto_c_exp = proto_c.expand(feats.size(0), feats.size(1))  # 扩展为 (x, feature_dim)

        # 拼接所有输入张量
        input_tensor = torch.cat([feats, prev_feats, proto_g_exp, proto_c_exp], dim=1)  # 形状为 (x, 2048)

        # 使用 MLP 生成新的特征
        x = self.fc1(input_tensor)  # 第一层
        x = self.relu(x)  # 激活
        output = self.fc2(x)  # 第二层

        return output

class FedPDAv2Client(FedAvgClient):
    def __init__(self, **commons):
        super().__init__(**commons)
        self.global_prototypes = [None] * NUM_CLASSES[self.args.dataset.name]
        self._proto_ready = False
        self.prev_model = deepcopy(self.model)

        feat_dim = self.model.get_last_features(
            self.dataset[0][0].unsqueeze(0).to(self.device)
        ).shape[1]

        self.generator = self._get_generator(feat_dim, commons)
        if self.generator:
            self.gen_opt = torch.optim.SGD(self.generator.parameters(), lr=self.args.fedpdav2.gen_lr)

        self.gen_mult = self.args.fedpdav2.gen_mult
        self.noise_std = self.args.fedpdav2.noise_std
        self.gen_interval = self.args.fedpdav2.gen_interval
        self.temperature = self.args.fedpdav2.temperature

    def _get_generator(self, feat_dim, commons):
        if commons["args"].fedpdav2.generator == "linear":
            return LinearGenerator().to(self.device)
        elif commons["args"].fedpdav2.generator == "nonlinear":
            return NonlinearGenerator(feature_dim=feat_dim).to(self.device)
        return None

    def set_parameters(self, package: dict[str, Any]):
        super().set_parameters(package)
        num_classes = NUM_CLASSES[self.args.dataset.name]
        gp = package.get("global_prototypes")
        self.current_epoch = package.get("current_epoch")

        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])
        else:
            self.prev_model.load_state_dict(self.model.state_dict())

        self.global_prototypes = [gp.get(c, None) if isinstance(gp, dict) else (gp + [None] * num_classes)[c] for c in
                                  range(num_classes)]
        self._valid_proto_idx = [i for i, p in enumerate(self.global_prototypes) if isinstance(p, torch.Tensor)]
        self._proto_ready = len(self._valid_proto_idx) > 0

        self.overwrite_classifier()
        self.align_federated_parameters()

    def package(self) -> dict[str, Any]:
        pkg = super().package()
        pkg["prototypes"] = self._compute_local_prototypes(mean=True)
        pkg["prev_model_state"] = deepcopy(self.model.state_dict())
        return pkg

    def overwrite_classifier(self):
        with torch.no_grad():
            prev_sd, curr_sd = self.prev_model.classifier.state_dict(), self.model.classifier.state_dict()
            diff_tensors = {k: (curr_sd[k] - prev_sd[k]).abs() for k in curr_sd}
            all_diff = torch.cat([v.flatten() for v in diff_tensors.values()])
            tau = torch.topk(all_diff, max(int(all_diff.numel() * 1), 1), largest=True).values.min()

            for k in curr_sd.keys():
                curr_sd[k][diff_tensors[k] >= tau] = prev_sd[k][diff_tensors[k] >= tau]

            self.model.classifier.load_state_dict(curr_sd, strict=False)

    @torch.no_grad()
    def _compute_local_prototypes(self, mean: bool = False):
        num_classes = NUM_CLASSES[self.args.dataset.name]
        self.protos = [[] for _ in range(num_classes)]
        self.model.eval()

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            self.feats = self.model.get_last_features(x)
            for f, lbl in zip(self.feats, y):
                self.protos[lbl.item()].append(f.detach())

        for c in range(num_classes):
            if self.protos[c]:
                self.protos[c] = torch.stack(self.protos[c])
                if mean:
                    self.protos[c] = self.protos[c].mean(dim=0)

        # if self.client_id == 81 and self.args.dataset.name == 'cifar10':
        #     self.visualize_prototypes(self.protos, self.global_prototypes)

        return self.protos

    def _loss_align(self, feats, labels):
        if self.testing or not self._proto_ready:
            return torch.tensor(0.0, device=self.device)
        loss = 0.0
        for c, proto in enumerate(self.global_prototypes):
            if proto is None:
                continue
            mask = (labels == c)
            if mask.any():
                loss += F.mse_loss(feats[mask].mean(0), proto.to(self.device))
        return self.args.fedpdav2.lambda_align * loss

    def _loss_contrast(self, feats, labels, temperature):
        if self.testing or not self._proto_ready:
            return torch.tensor(0.0, device=self.device)

        idx = self._valid_proto_idx
        proto_stack = torch.stack([self.global_prototypes[i] for i in idx]).to(self.device)  # [K, D]
        feats = F.normalize(feats, dim=1)
        proto_stack = F.normalize(proto_stack, dim=1)
        logits = feats @ proto_stack.T / temperature  # [B, K]

        # 仅计算那些标签在 idx 子集内的样本
        keep_mask = torch.zeros_like(labels, dtype=torch.bool)
        remap = {c: i for i, c in enumerate(idx)}
        mapped = labels.clone()
        for c, i in remap.items():
            m = (labels == c)
            keep_mask |= m
            mapped[m] = i
        if not keep_mask.any():
            return torch.tensor(0.0, device=self.device)

        return self.args.fedpdav2.lambda_proto * F.cross_entropy(logits[keep_mask], mapped[keep_mask])

    def _maybe_update_generator(self, feats_c: torch.Tensor, prev_c: torch.Tensor,
                                labels_c: torch.Tensor, proto_g: torch.Tensor, proto_c: torch.Tensor):
        """
        feats_c/prev_c: [Bc, D]，只对应某一个类 c
        labels_c: [Bc]，值全是该类的 id
        proto_g/proto_c: [1, D]
        """
        if self.testing or self.generator is None or not self._proto_ready:
            return
        if (self.training_step % self.args.fedpdav2.gen_interval):  # 仅当==0时更新
            return

        # 1) 生成合成特征（不 detach，允许梯度回传到生成器）
        synth_feat = self.generator(feats_c, prev_c, proto_g, proto_c)  # [Bc, D]

        # 2) 冻结 classifier，仅让生成器学习
        cls_params = list(self.model.classifier.parameters())
        requires_backup = [p.requires_grad for p in cls_params]
        for p in cls_params:
            p.requires_grad_(False)

        # 3) 前向并计算生成器的监督损失
        logits = self.model.classifier(synth_feat)  # [Bc, num_classes]
        gen_loss = self.criterion(logits, labels_c)

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

        # 4) 恢复 classifier 的 requires_grad
        for p, rg in zip(cls_params, requires_backup):
            p.requires_grad_(rg)

    def fit(self):
        """
        进阶版：
          - 对每个在当前 batch 出现的类别 c，若有本地/全局原型，则一次性生成 Bc * gen_mult 个合成特征；
          - 生成器的更新在“按类循环”里完成，只用该类的数据/标签/原型，避免错配；
          - 分类器训练阶段对合成特征 detach，避免梯度回到生成器；
          - 可选噪声增强：args.fedpdav2.noise_std（默认 0）；
          - 倍增系数：args.fedpdav2.gen_mult（默认 1）。
        """
        self.prev_model.eval()
        self.model.train()
        self.dataset.train()
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.args.optimizer.lr, momentum=0.9)

        self.training_step = 0
        for _ in range(self.local_epoch):
            # 预先计算一次本地原型（均值）
            self._compute_local_prototypes(mean=True)

            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)

                # 前向（真实样本）
                logits = self.model(x)
                self.feats = self.model.get_last_features(x, detach=False)  # [B, D]
                self.prev_feats = self.prev_model.get_last_features(x, detach=False)

                # 基础损失 + 对齐/对比损失（只用真实样本）
                loss = self.criterion(logits, y)
                loss = loss + self._loss_align(self.feats, y)
                loss = loss + self._loss_contrast(self.feats, y, self.temperature)

                # ===== 合成样本生成 & 生成器更新（按类） =====
                all_synth_feats = []
                all_labels = []

                if (self.generator is not None) and self._proto_ready:
                    for label in y.unique():
                        lid = int(label.item())

                        # 本地/全局原型（都要存在）
                        local_proto = None
                        if lid < len(self.protos) and isinstance(self.protos[lid], torch.Tensor):
                            local_proto = self.protos[lid].to(self.device)  # [D]

                        global_proto = None
                        if lid < len(self.global_prototypes) and isinstance(self.global_prototypes[lid], torch.Tensor):
                            global_proto = self.global_prototypes[lid].to(self.device)  # [D]

                        if (local_proto is None) or (global_proto is None):
                            continue

                        # 该类在本 batch 的真实特征
                        mask_c = (y == lid)
                        feats_c = self.feats[mask_c].detach()  # [Bc, D]
                        prev_c = self.prev_feats[mask_c].detach()  # [Bc, D]
                        if feats_c.numel() == 0:
                            continue

                        # 倍增（k = gen_mult），可选噪声增强
                        if self.gen_mult > 1:
                            feats_in = feats_c.repeat(self.gen_mult, 1)
                            prev_in = prev_c.repeat(self.gen_mult, 1)
                        else:
                            feats_in = feats_c
                            prev_in = prev_c

                        if self.noise_std and self.noise_std > 0:
                            feats_in = feats_in + torch.randn_like(feats_in) * float(self.noise_std)

                        # 为生成器提供条件（[1, D]）
                        proto_c = local_proto.unsqueeze(0)
                        proto_g = global_proto.unsqueeze(0)

                        # —— 生成“该类”的合成特征（不 detach：用于训练生成器）——
                        synth_feat_c = self.generator(feats_in, prev_in, proto_g, proto_c)  # [k*Bc, D]
                        labels_c = torch.full((synth_feat_c.size(0),), lid,
                                              device=self.device, dtype=y.dtype)

                        # —— 按类更新生成器（周期控制）——
                        if (self.gen_opt is not None) and (self.training_step % self.gen_interval == 0):
                            # 暂时冻结 classifier，仅让生成器学
                            cls_params = list(self.model.classifier.parameters())
                            requires_backup = [p.requires_grad for p in cls_params]
                            for p in cls_params:
                                p.requires_grad_(False)

                            logits_gen = self.model.classifier(synth_feat_c)  # 不 detach
                            gen_loss = self.criterion(logits_gen, labels_c)

                            self.gen_opt.zero_grad()
                            gen_loss.backward()
                            self.gen_opt.step()

                            # 恢复 classifier 的 requires_grad
                            for p, rg in zip(cls_params, requires_backup):
                                p.requires_grad_(rg)

                        # 分类器训练用的合成样本要 detach，避免梯度回流到生成器
                        all_synth_feats.append(synth_feat_c.detach())
                        all_labels.append(labels_c)

                # ===== 拼接真实 + 合成，一起训练分类器 =====
                if all_synth_feats:
                    synth_feats = torch.cat(all_synth_feats, dim=0)  # [S, D]
                    labels_synth = torch.cat(all_labels, dim=0)  # [S]
                    logits_synth = self.model.classifier(synth_feats)  # 仅分类器前向

                    logits_all = torch.cat([logits, logits_synth], dim=0)
                    y_all = torch.cat([y, labels_synth], dim=0)

                    # 重新计算分类器损失（包含合成样本），并叠加对齐/对比（仍只用真实特征）
                    loss_cls = self.criterion(logits_all, y_all)
                    loss = loss_cls + self._loss_align(self.feats, y) + self._loss_contrast(self.feats, y, self.temperature)

                # ===== 客户端模型更新 =====
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                self.training_step += 1

            # epoch 末尾：刷新一次本地原型与 prev_model
            self._compute_local_prototypes(mean=True)
            self.prev_model = deepcopy(self.model)

    def visualize_prototypes(self, local_protos, global_protos, class_names=None, save_path="proto_cifar10.pdf"):
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA

        data, classes, types = [], [], []
        for cls, p in enumerate(local_protos + global_protos):
            if isinstance(p, torch.Tensor):
                data.append(p.cpu().numpy())
                classes.append(cls)
                types.append("Local" if cls < len(local_protos) else "Global")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(data))
        df = pd.DataFrame({"x": reduced[:, 0], "y": reduced[:, 1], "Class": classes, "Type": types})
        self._plot_prototypes(df, class_names, save_path)

    def _plot_prototypes(self, df, class_names, save_path):
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_title(f"CIFAR10 (Before)", fontsize=22, weight="bold")
        ax.set_xlabel("PC-1", fontsize=20, weight="bold")
        ax.set_ylabel("PC-2", fontsize=20, weight="bold")
        ax.grid(True, linestyle='--', alpha=0.5)

        unique_classes = df["Class"].unique()
        color_map = plt.get_cmap("tab10")

        for i, cls in enumerate(unique_classes):
            color = color_map(i % 10)
            group_local = df[(df["Class"] == cls) & (df["Type"] == "Local")]
            group_global = df[(df["Class"] == cls) & (df["Type"] == "Global")]

            if not group_local.empty:
                ax.scatter(group_local["x"], group_local["y"], marker='o', s=100, c=[color], edgecolors='black',
                           label="Local" if i == 0 else "", zorder=3)
            if not group_global.empty:
                ax.scatter(group_global["x"], group_global["y"], marker='X', s=100, c=[color], edgecolors='black',
                           label="Global" if i == 0 else "", zorder=3)

            if not group_local.empty and not group_global.empty:
                ax.add_patch(FancyArrowPatch(
                    posA=(group_local["x"].values[0], group_local["y"].values[0]),
                    posB=(group_global["x"].values[0], group_global["y"].values[0]),
                    arrowstyle="-|>", linestyle="solid", color="limegreen", mutation_scale=12, lw=1.5, alpha=0.9,
                    zorder=2
                ))

        legend = ax.legend(loc="lower right", title="Type", frameon=True, framealpha=0.6, fontsize=20)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.8)
        for text in legend.get_texts():
            text.set_fontweight("bold")

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {save_path}")

    def align_federated_parameters(self):
        self.prev_model.eval()
        self.prev_model.to(self.device)
        self.model.train()
        self.dataset.train()

        prototypes = [[] for _ in range(NUM_CLASSES[self.args.dataset.name])]
        with torch.no_grad():
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                prev_feats = self.prev_model.get_last_features(x)

                for y, feat in zip(y, prev_feats):
                    prototypes[y].append(feat)

        mean_prototypes = [torch.stack(p).mean(dim=0) if p else None for p in prototypes]

        alignment_optimizer = torch.optim.SGD(self.model.base.parameters(), lr=0.01)

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            features = self.model.get_last_features(x, detach=False)
            loss = sum(
                F.mse_loss(features[y == label].mean(dim=0), mean_prototypes[label]) for label in y.unique().tolist() if
                mean_prototypes[label] is not None)

            alignment_optimizer.zero_grad()
            loss.backward()
            alignment_optimizer.step()