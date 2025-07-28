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


# ─────────────────────────  生成器模块  ──────────────────────────
class LinearGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4))  # feats, proto

    def forward(self, feats, prev_feats, proto_g, proto_c):
        w = F.softmax(self.w, dim=0)
        return w[0] * feats + w[1] * prev_feats + w[2] * proto_g + w[3] * proto_c


class NonlinearGenerator(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, feats, prev_feats, proto_g, proto_c):
        return self.mlp(torch.cat([feats, prev_feats, proto_g, proto_c], dim=1))


# ────────────────────────  FedPDA‑v2 Client  ────────────────────────
class FedPDAv2Client(FedAvgClient):
    """
    FedPDA‑v2：

    1. Prototype Alignment (MSE) w.r.t. global prototypes
    2. Prototype Contrastive Margin Loss
    3. Optional Generator‑based feature augmentation

    - 聚合权重仍为 len(trainset)
    - 不做分类器个性化
    """

    # ----------------------------- 初始化 ----------------------------- #
    def __init__(self, **commons):
        super().__init__(**commons)

        # 全局原型（list 长度 = C，若缺失则为 None）
        self.global_prototypes: List[torch.Tensor | None] = []
        self._proto_ready: bool = False  # 全局原型是否全部就绪
        self.prev_model = deepcopy(self.model)

        # 生成器（可选）
        feat_dim = self.model.get_last_features(
            self.dataset[0][0].unsqueeze(0).to(self.device)
        ).shape[1]

        if commons["args"].fedpdav2.generator == "linear":
            self.generator = LinearGenerator().to(self.device)
        elif commons["args"].fedpdav2.generator == "nonlinear":
            self.generator = NonlinearGenerator(feature_dim=feat_dim).to(self.device)
        else:
            self.generator = None

        if self.generator is not None:
            self.gen_opt = torch.optim.SGD(
                self.generator.parameters(), lr=self.args.fedpdav2.gen_lr
            )

    # ------------------------- 通信：server → client ------------------------- #
    def set_parameters(self, package: dict[str, Any]):
        """在每轮开始或测试时调用。这里把 global_prototypes 规整成定长 list。"""
        super().set_parameters(package)

        num_classes = NUM_CLASSES[self.args.dataset.name]
        gp = package.get("global_prototypes")

        if package["prev_model_state"] is not None:
            self.prev_model.load_state_dict(package["prev_model_state"])

        if isinstance(gp, dict):
            self.global_prototypes = [gp.get(c, None) for c in range(num_classes)]
        elif isinstance(gp, list):
            # 若 server 已给定定长 list，则直接使用；若不足补 None
            self.global_prototypes = (gp + [None] * num_classes)[:num_classes]
        else:
            self.global_prototypes = [None] * num_classes

        self._proto_ready = all(isinstance(p, torch.Tensor) for p in self.global_prototypes)
        self.overwrite_top1pct_classifier()

        if self.testing:
            self.model.load_state_dict(self.prev_model.state_dict())

    # ------------------------- client → server ------------------------- #
    def package(self) -> dict[str, Any]:
        pkg = super().package()
        pkg["prototypes"] = self._compute_local_prototypes(mean=True)
        pkg["prev_model_state"] = deepcopy(self.model.state_dict())
        return pkg

    def overwrite_top1pct_classifier(self):
        """仅把差值最大的 1% classifier 权重替换为 prev_model 的值"""
        with torch.no_grad():
            prev_sd = self.prev_model.classifier.state_dict()
            curr_sd = self.model.classifier.state_dict()

            # 1. 计算逐元素绝对差
            diff_tensors = {k: (curr_sd[k] - prev_sd[k]).abs() for k in curr_sd}

            # 2. 全量展开，找到阈值 τ（Top‑1%）
            all_diff = torch.cat([v.flatten() for v in diff_tensors.values()])
            k = max(int(all_diff.numel() * 0.05), 1)  # 至少取 1 个元素
            tau = torch.topk(all_diff, k, largest=True).values.min()

            # 3. 逐张量覆盖差值 ≥ τ 的位置
            for k in curr_sd.keys():
                mask = diff_tensors[k] >= tau  # Bool 同形状
                curr_sd[k][mask] = prev_sd[k][mask]

            # 4. 写回 classifier
            self.model.classifier.load_state_dict(curr_sd, strict=False)

    # ----------------------- 原型相关辅助函数 ----------------------- #
    @torch.no_grad()
    def _compute_local_prototypes(self, mean: bool = False):
        """收集本地每类特征；返回 list[len=C]"""
        num_classes = NUM_CLASSES[self.args.dataset.name]
        self.protos: List[list | torch.Tensor] = [[] for _ in range(num_classes)]

        self.model.eval()
        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            feats = self.model.get_last_features(x)
            for f, lbl in zip(feats, y):
                self.protos[lbl.item()].append(f.detach())

        for c in range(num_classes):
            if len(self.protos[c]) > 0:
                self.protos[c] = torch.stack(self.protos[c])
                if mean:
                    self.protos[c] = self.protos[c].mean(dim=0)
        # if self.client_id == 81 and self.args.dataset.name == 'cifar10':
        #     self.visualize_prototypes(self.protos, self.global_prototypes)
        return self.protos

    # ---------- 安全版 Prototype Alignment（测试或未就绪时返回 0） ---------- #
    def _loss_align(self, feats, labels):
        if self.testing or not self._proto_ready:
            return torch.tensor(0.0, device=self.device)

        loss = 0.0
        for c in labels.unique():
            proto = self.global_prototypes[c.item()]
            if proto is None:
                continue
            idx = labels == c
            loss += F.mse_loss(feats[idx].mean(0), proto.to(self.device))
        return self.args.fedpdav2.lambda_align * loss

    # ---------- 安全版 Prototype Contrastive Loss ---------- #
    def _loss_contrast(self, feats, labels, temperature=0.5):
        if self.testing or not self._proto_ready:
            return torch.tensor(0.0, device=self.device)

        proto_stack = torch.stack(self.global_prototypes).to(self.device)
        feats = F.normalize(feats, dim=1)
        proto_stack = F.normalize(proto_stack, dim=1)

        logits = feats @ proto_stack.T              # B × C
        logits /= temperature

        loss = F.cross_entropy(logits, labels)
        return self.args.fedpdav2.lambda_proto * loss

    # ---------- 生成器更新（只有在训练且原型就绪时才执行） ---------- #
    def _maybe_update_generator(self, feats_detached, labels):
        if (
                self.testing
                or self.generator is None
                or not self._proto_ready
                or self.training_step % self.args.fedpdav2.gen_interval
        ):
            return

        synth_feat = self.generator(
            feats_detached,
            self.prev_feats,
            self.proto_global,
            self.proto_global,
        )

        logits_synth = self.model.classifier(synth_feat)
        gen_loss = self.criterion(logits_synth, labels)

        self.gen_opt.zero_grad()
        gen_loss.backward()
        self.gen_opt.step()

    # ----------------------------- 训练主函数 ----------------------------- #
    def fit(self):
        """重写 fit，但保持 FedAvg 流程；所有新特性通过安全函数注入。"""
        self._compute_local_prototypes(mean=True)
        self.prev_model.eval()
        self.model.train()
        self.dataset.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.args.optimizer.lr, momentum=0.9
        )

        self.training_step = 0
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                self.feats = self.model.get_last_features(x, detach=False)
                self.prev_feats = self.prev_model.get_last_features(x, detach=False)

                # 基本 loss
                loss = self.criterion(logits, y)
                loss += self._loss_align(self.feats, y)
                loss += self._loss_contrast(self.feats, y, self.args.fedpdav2.temperature)

                # 如果启用 generator，生成额外样本并合并
                if self.generator is not None and self._proto_ready:
                    self.proto_global = torch.stack([self.global_prototypes[l.item()] for l in y]).to(self.device)
                    self.proto_current = torch.stack([self.protos[l.item()] for l in y]).to(self.device)
                    synth_feat = self.generator(self.feats.detach(), self.prev_feats.detach(), self.proto_global, self.proto_current)
                    logits_synth = self.model.classifier(synth_feat)

                    # 合并真实 + 合成 logits；loss 不变
                    logits_all = torch.cat([logits, logits_synth], dim=0)
                    y_all = torch.cat([y, y], dim=0)
                    loss = self.criterion(logits_all, y_all) \
                           + self._loss_align(self.feats, y) \
                           + self._loss_contrast(self.feats, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

                # 生成器（可选）
                self._maybe_update_generator(self.feats.detach(), y)
                self.training_step += 1

    def visualize_prototypes(
        self, local_protos: List[torch.Tensor | None],
        global_protos: List[torch.Tensor | None],
        class_names: List[str] = None,
        save_path: str = f"proto_cifar10.pdf",
    ):
        """
        使用纯 matplotlib 实现 prototype 可视化（local vs global），
        支持 Times New Roman，图例在图内右下角，箭头为亮绿色，类别颜色保持。
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        from matplotlib.patches import FancyArrowPatch

        # 1. 准备原型数据
        data = []
        classes = []
        types = []

        for cls, p in enumerate(local_protos):
            if isinstance(p, torch.Tensor):
                data.append(p.cpu().numpy())
                classes.append(cls)
                types.append("Local")

        for cls, p in enumerate(global_protos):
            if isinstance(p, torch.Tensor):
                data.append(p.cpu().numpy())
                classes.append(cls)
                types.append("Global")

        data = np.array(data)
        labels = [class_names[i] if class_names else str(i) for i in classes]

        # 2. 降维
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)

        # 3. 构建 DataFrame
        df = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "Class": labels,
            "Type": types
        })

        # 4. 可视化
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        ax.set_title(f"CIFAR10 (Before)", fontsize=22, weight="bold")
        ax.set_xlabel("PC-1", fontsize=20, weight="bold")
        ax.set_ylabel("PC-2", fontsize=20, weight="bold")
        ax.grid(True, linestyle='--', alpha=0.5)

        unique_classes = df["Class"].unique()
        color_map = plt.get_cmap("tab10")

        # 5. 绘制点
        for i, cls in enumerate(unique_classes):
            color = color_map(i % 10)
            group_local = df[(df["Class"] == cls) & (df["Type"] == "Local")]
            group_global = df[(df["Class"] == cls) & (df["Type"] == "Global")]

            if not group_local.empty:
                ax.scatter(group_local["x"], group_local["y"], marker='o', s=100, c=[color], edgecolors='black', label="Local" if i == 0 else "", zorder=3)
            if not group_global.empty:
                ax.scatter(group_global["x"], group_global["y"], marker='X', s=100, c=[color], edgecolors='black', label="Global" if i == 0 else "", zorder=3)

            # 6. 添加箭头
            if not group_local.empty and not group_global.empty:
                arrow = FancyArrowPatch(
                    posA=(group_local["x"].values[0], group_local["y"].values[0]),
                    posB=(group_global["x"].values[0], group_global["y"].values[0]),
                    arrowstyle="-|>",
                    linestyle="solid",
                    color="limegreen",
                    mutation_scale=12,
                    lw=1.5,
                    alpha=0.9,
                    zorder=2
                )
                ax.add_patch(arrow)

        # 7. 设置图例在图内右下角，半透明
        legend = ax.legend(
            loc="lower right",
            title="Type",
            frameon=True,
            framealpha=0.6,
            fontsize=20,
            title_fontsize=20,
            borderpad=0.8,
            labelspacing=0.4
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(0.8)
        for text in legend.get_texts():
            text.set_fontweight("bold")

        legend.get_title().set_fontweight("bold")

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {save_path}")