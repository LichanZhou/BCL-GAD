import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import BertTokenizer, BertModel
import scipy.stats as stats
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.optim import AdamW

# -------------------------- 数据预处理：多模态图数据加载 --------------------------
class MultimodalGraphDataset:
    def __init__(self, graph_data, text_data, attr_data, is_source_domain=True):
        """
        graph_data: PyG Data对象（含edge_index、x拓扑特征）
        text_data: 文本列表（如交易备注）
        attr_data: 属性矩阵（如账户信用分）
        """
        self.graph_data = graph_data
        self.text_data = text_data
        self.attr_data = StandardScaler().fit_transform(attr_data)  # 归一化
        self.is_source = is_source_domain

        # 文本编码（用BERT提取文本特征）
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').eval()
        self.text_features = self.encode_text()

    def encode_text(self):
        """文本模态特征编码"""
        with torch.no_grad():
            inputs = self.tokenizer(self.text_data, padding=True, truncation=True, return_tensors='pt', max_length=16)
            outputs = self.bert(**inputs)
            return outputs.last_hidden_state[:, 0, :].numpy()  # [num_nodes, 768]


# -------------------------- 模态权重计算+对比对生成 --------------------------
def get_modal_weights_and_pairs(dataset, cmi_threshold=0.08):
    """
    输入：纯净数据集（因果去噪后）
    输出：模态权重 + 正负对比对
    """
    num_nodes = len(dataset.graph_data.x)
    # 提取三模态特征（均转为numpy数组，便于计算CMI）
    topo_feat = dataset.graph_data.x.numpy()  # [num_nodes, topo_dim]
    text_feat = dataset.text_features  # [num_nodes, 768]
    attr_feat = dataset.attr_data  # [num_nodes, attr_dim]

    all_feat = np.hstack([topo_feat, text_feat, attr_feat])  # [num_nodes, 总维度]
    nn_model = NearestNeighbors(n_neighbors=min(11, num_nodes), metric='euclidean')
    nn_model.fit(all_feat)
    _, neighbor_indices = nn_model.kneighbors(all_feat)  # [num_nodes, k+1]（包含自身）
    # ------------------------------------------------------------------------------

    node_cmi = []
    for i in range(num_nodes):
        # ---------------------- 修改：用邻域样本替代单节点样本 ----------------------
        neighbor_idx = neighbor_indices[i]  # 当前节点的邻域索引（自身+k个近邻）
        # 提取邻域的高维特征
        neighbor_text = text_feat[neighbor_idx]  # [k+1, 768]
        neighbor_topo = topo_feat[neighbor_idx]  # [k+1, topo_dim]
        neighbor_attr = attr_feat[neighbor_idx]  # [k+1, attr_dim]
        # PCA降维到1维（适配CMI函数的一维输入）
        x = pca_to_1dim(neighbor_text)  # 文本X：[k+1,]（一维数组）
        y = pca_to_1dim(neighbor_topo)  # 拓扑Y：[k+1,]（一维数组）
        z = pca_to_1dim(neighbor_attr)  # 属性Z：[k+1,]（一维数组）
        # 计算3组CMI（简化：用特征向量的均值降维，实际可用PCA降维后计算）
        # 计算3组CMI（简化：用特征向量的均值降维，实际可用PCA降维后计算）
        cmi_xy_z = conditional_mutual_info(x, y, z)
        cmi_xz_y = conditional_mutual_info(x, z, y)
        cmi_yz_x = conditional_mutual_info(y, z, x)
        node_cmi.append([cmi_xy_z, cmi_xz_y, cmi_yz_x])
    node_cmi = np.array(node_cmi)  # [num_nodes, 3]

    # 步骤2：筛选纯净正常样本（3组CMI均≥阈值）
    normal_nodes = np.where((node_cmi[:, 0] >= cmi_threshold) &
                            (node_cmi[:, 1] >= cmi_threshold) &
                            (node_cmi[:, 2] >= cmi_threshold))[0]
    abnormal_nodes = np.setdiff1d(np.arange(num_nodes), normal_nodes)  # 异常样本（含虚假关联）

    # 步骤3：计算全局模态权重（基于正常样本的CMI求和）
    total_cmi = node_cmi[normal_nodes].sum(axis=0)  # [3]
    modal_weights = total_cmi / total_cmi.sum()  # 归一化 [w_text, w_topo, w_attr]（对应CMI的两组关联和）
    modal_weights = torch.tensor(modal_weights, dtype=torch.float32)

    # 步骤4：生成对比对（正向：正常样本互配；负向：正常-异常样本互配）
    positive_pairs = []
    for i in range(len(normal_nodes)):
        for j in range(i + 1, len(normal_nodes)):
            positive_pairs.append((normal_nodes[i], normal_nodes[j]))

    negative_pairs = []
    for i in normal_nodes:
        for j in abnormal_nodes[:10]:  # 每个正常样本配10个异常样本（避免负样本过多）
            negative_pairs.append((i, j))

    return modal_weights, positive_pairs, negative_pairs, normal_nodes


# -------------------------- 多模态GNN编码器 --------------------------
class MultimodalGNNEncoder(nn.Module):
    def __init__(self, text_dim=768, topo_dim=10, attr_dim=5, hidden_dim=256, out_dim=128):
        super().__init__()
        # 单模态编码器
        self.topo_encoder = GATConv(topo_dim, hidden_dim)  # 拓扑模态（GAT）
        self.attr_encoder = nn.Sequential(nn.Linear(attr_dim, hidden_dim), nn.ReLU())  # 属性模态（MLP）
        self.text_encoder = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.ReLU())  # 文本模态（MLP，基于BERT输出）

        # 融合层
        self.fusion = nn.Linear(hidden_dim * 3, out_dim)

    def forward(self, topo_x, edge_index, attr_x, text_x):
        """
        输入：各模态特征
        输出：单模态特征 + 加权融合特征
        """
        # 单模态编码
        topo_feat = F.relu(self.topo_encoder(topo_x, edge_index))  # [num_nodes, hidden_dim]
        attr_feat = self.attr_encoder(attr_x)  # [num_nodes, hidden_dim]
        text_feat = self.text_encoder(text_x)  # [num_nodes, hidden_dim]

        return text_feat, topo_feat, attr_feat


# -------------------------- 对比学习损失（加权InfoNCE） --------------------------
class WeightedInfoNCE(nn.Module):
    def __init__(self, temperature=0.07,max_neg_samples=200):
        super().__init__()
        self.temp = temperature
        self.max_neg_samples = max_neg_samples

    def forward(self, modal_weights, text_feat, topo_feat, attr_feat, pos_pairs, neg_pairs):
        """
        modal_weights: [w_text, w_topo, w_attr]
        pos_pairs/neg_pairs: 对比对列表（(i,j)）
        """

        # 生成批次对比对特征
        def get_pair_feats(feats, pairs):
            if not pairs:
                return torch.empty(0, feats.shape[1]).to(feats.device), torch.empty(0, feats.shape[1]).to(feats.device)
            i_indices = torch.tensor([p[0] for p in pairs], dtype=torch.long).to(feats.device)
            j_indices = torch.tensor([p[1] for p in pairs], dtype=torch.long).to(feats.device)
            i_feats = feats[i_indices]
            j_feats = feats[j_indices]
            return i_feats, j_feats

        # 单模态对比损失（修复核心：正确实现InfoNCE）
        def modal_contrast_loss(feats, pos_pairs, neg_pairs):
            pos_anchor, pos_sample = get_pair_feats(feats, pos_pairs)
            neg_anchor, neg_sample = get_pair_feats(feats, neg_pairs)
            device = feats.device

            # 边界处理：正负样本都不能为空，且数量不能为0
            if pos_anchor.shape[0] == 0 or neg_anchor.shape[0] == 0:
                return torch.tensor(0.0, requires_grad=True).to(device)

            # 正常场景：有正有负 → 计算InfoNCE
            feats = F.normalize(feats, p=2, dim=1)
            pos_anchor = F.normalize(pos_anchor, p=2, dim=1)
            pos_sample = F.normalize(pos_sample, p=2, dim=1)
            neg_anchor = F.normalize(neg_anchor, p=2, dim=1)
            neg_sample = F.normalize(neg_sample, p=2, dim=1)

            # 核心修复1：限制负样本数量（避免维度爆炸）
            num_neg = min(neg_sample.shape[0], self.max_neg_samples)
            if num_neg < neg_sample.shape[0]:
                # 随机采样num_neg个负样本（保证训练随机性）
                neg_idx = torch.randperm(neg_sample.shape[0])[:num_neg].to(device)
                neg_anchor = neg_anchor[neg_idx]
                neg_sample = neg_sample[neg_idx]

            # 核心修复：计算neg_sim后转置为 [1, num_neg]（singleton维度0）
            pos_sim = F.cosine_similarity(pos_anchor, pos_sample, dim=1).unsqueeze(1) / self.temp  # [N_pos, 1]
            neg_sim = F.cosine_similarity(neg_anchor, neg_sample, dim=1).unsqueeze(1) / self.temp  # [num_neg, 1]
            neg_sim = neg_sim.T  # 转置后：[1, num_neg]（维度0是singleton，可expand）

            # 构建logits（每个正样本对应所有负样本）
            # 核心修复：用repeat复制负样本，适配正样本数量（每个正样本对应所有负样本）
            # repeat(a, b)：对第0维复制a次，第1维复制b次（-1表示保持原维度）
            logits = torch.cat([pos_sim, neg_sim.expand(pos_sim.shape[0], -1)], dim=1)  # [N_pos, 1 + num_neg]
            labels = torch.zeros(pos_sim.shape[0], dtype=torch.long).to(device)

            # 计算交叉熵损失（返回可求导的张量）
            loss = F.cross_entropy(logits, labels,label_smoothing=0.1)
            return loss

        # 计算各模态损失（此时所有损失都是张量，无None）
        text_loss = modal_contrast_loss(text_feat, pos_pairs, neg_pairs)
        topo_loss = modal_contrast_loss(topo_feat, pos_pairs, neg_pairs)
        attr_loss = modal_contrast_loss(attr_feat, pos_pairs, neg_pairs)

        # 融合特征损失
        fused_feat = modal_weights[0] * text_feat + modal_weights[1] * topo_feat + modal_weights[2] * attr_feat
        fused_loss = modal_contrast_loss(fused_feat, pos_pairs, neg_pairs)

        # 加权总损失（张量求和，仍为可求导张量）
        total_loss = (modal_weights[0] * text_loss +
                      modal_weights[1] * topo_loss +
                      modal_weights[2] * attr_loss +
                      fused_loss)
        return total_loss


# -------------------------- 源域预训练（多模态因果调整对比对） --------------------------
def source_pretrain(encoder, source_dataset, modal_weights, pos_pairs, neg_pairs, epochs=50, lr=1e-4):
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-5)
    criterion = WeightedInfoNCE(temperature=0.07)
    encoder.train()

    # 转换数据为tensor
    topo_x = source_dataset.graph_data.x.float()
    edge_index = source_dataset.graph_data.edge_index
    attr_x = torch.tensor(source_dataset.attr_data, dtype=torch.float32)
    text_x = torch.tensor(source_dataset.text_features, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        # 编码得到各模态特征
        text_feat, topo_feat, attr_feat = encoder(topo_x, edge_index, attr_x, text_x)
        # 计算加权对比损失
        loss = criterion(modal_weights, text_feat, topo_feat, attr_feat, pos_pairs, neg_pairs)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Source Pretrain Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return encoder


# -------------------------- 跨域微调核心逻辑 --------------------------
def cross_domain_finetune(pretrained_encoder, source_dataset, target_dataset, modal_weights, normal_nodes, epochs=20,
                          lr=5e-5, weight_lr=1e-4):  # 单独给权重设置学习率（更灵活）
    """
    核心修改：
    1. 动态冻结：残差小的模态→全冻结；残差大的模态→只冻底层、微调顶层
    2. 权重可训练：将模态权重设为nn.Parameter，支持反向传播更新
    3. 闭环优化：权重与编码器参数一起通过损失函数优化，而非手动调整
    """
    # 步骤1：构建源域正常原型（基于预训练编码器）
    pretrained_encoder.eval()
    with torch.no_grad():
        source_topo_x = source_dataset.graph_data.x.float()
        source_edge_index = source_dataset.graph_data.edge_index
        source_attr_x = torch.tensor(source_dataset.attr_data, dtype=torch.float32)
        source_text_x = torch.tensor(source_dataset.text_features, dtype=torch.float32)

        source_text_feat, source_topo_feat, source_attr_feat = pretrained_encoder(
            source_topo_x, source_edge_index, source_attr_x, source_text_x
        )
        # 源域正常原型（各模态均值）
        source_normal_text_proto = source_text_feat[normal_nodes].mean(dim=0)
        source_normal_topo_proto = source_topo_feat[normal_nodes].mean(dim=0)
        source_normal_attr_proto = source_attr_feat[normal_nodes].mean(dim=0)
        source_protos = (source_normal_text_proto, source_normal_topo_proto, source_normal_attr_proto)

    # 步骤2：目标域数据准备
    target_topo_x = target_dataset.graph_data.x.float()
    target_edge_index = target_dataset.graph_data.edge_index
    target_attr_x = torch.tensor(target_dataset.attr_data, dtype=torch.float32)
    target_text_x = torch.tensor(target_dataset.text_features, dtype=torch.float32)
    # 目标域已知正常节点（示例：前50个，可替换为真实标注）
    target_normal_nodes = torch.tensor(np.arange(50), dtype=torch.long)

    # 步骤3：模态权重转为可训练参数（核心修改1：权重支持反向传播）
    trainable_weights = nn.Parameter(modal_weights.clone(), requires_grad=True)  # [w_text, w_topo, w_attr]

    # 步骤4：动态冻结编码器参数（核心修改2：按残差大小判断冻结策略）
    # 先计算初始残差（用于判断模态可靠性）
    pretrained_encoder.eval()
    with torch.no_grad():
        target_text_feat_init, target_topo_feat_init, target_attr_feat_init = pretrained_encoder(
            target_topo_x, target_edge_index, target_attr_x, target_text_x
        )
        # 初始单模态残差（目标域正常样本与源域原型的距离）
        text_res_init = F.pairwise_distance(target_text_feat_init[target_normal_nodes],
                                           source_protos[0].unsqueeze(0)).mean().item()
        topo_res_init = F.pairwise_distance(target_topo_feat_init[target_normal_nodes],
                                           source_protos[1].unsqueeze(0)).mean().item()
        attr_res_init = F.pairwise_distance(target_attr_feat_init[target_normal_nodes],
                                           source_protos[2].unsqueeze(0)).mean().item()
        init_residuals = [text_res_init, topo_res_init, attr_res_init]
        print(f"初始单模态残差：文本={text_res_init:.4f}, 拓扑={topo_res_init:.4f}, 属性={attr_res_init:.4f}")

    # 定义模态名称与对应编码器层（需与MultimodalGNNEncoder结构对应）
    modal_layers = {
        "text": pretrained_encoder.text_encoder,  # 文本编码器（MLP：Linear+ReLU）
        "topo": pretrained_encoder.topo_encoder,  # 拓扑编码器（GAT）
        "attr": pretrained_encoder.attr_encoder   # 属性编码器（MLP：Linear+ReLU）
    }
    residual_threshold = np.mean(init_residuals)  # 残差阈值（均值，可自定义）

    # 动态冻结逻辑：
    for idx, (modal_name, layers) in enumerate(modal_layers.items()):
        current_res = init_residuals[idx]
        if current_res < residual_threshold:
            # 残差小→模态在新域可靠，全冻结（不更新任何参数）
            for param in layers.parameters():
                param.requires_grad = False
            print(f"模态[{modal_name}]：残差小（{current_res:.4f} < {residual_threshold:.4f}）→ 全冻结")


    # 步骤5：优化器（同时优化编码器可训练参数+模态权重）
    params_to_optimize = [
        {"params": filter(lambda p: p.requires_grad, pretrained_encoder.parameters()), "lr": lr},
        {"params": [trainable_weights], "lr": weight_lr}  # 权重单独设置学习率
    ]
    optimizer = AdamW(params_to_optimize, weight_decay=1e-5)

    # 步骤6：迭代微调（权重+编码器参数联合优化）
    pretrained_encoder.train()
    target_normal_threshold = 1  # 融合残差目标阈值（基于源域统计）
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 6.1 目标域特征编码（用当前编码器和可训练权重）
        target_text_feat, target_topo_feat, target_attr_feat = pretrained_encoder(
            target_topo_x, target_edge_index, target_attr_x, target_text_x
        )

        # 6.2 归一化模态权重（避免权重之和不为1）
        normalized_weights = F.softmax(trainable_weights, dim=0)  # 确保权重∈[0,1]且和为1

        # 6.3 计算融合特征与融合残差（目标域正常样本）
        target_normal_text_feat = target_text_feat[target_normal_nodes]
        target_normal_topo_feat = target_topo_feat[target_normal_nodes]
        target_normal_attr_feat = target_attr_feat[target_normal_nodes]

        # 加权融合特征（用归一化后的可训练权重）
        fused_feat = (normalized_weights[0] * target_normal_text_feat +
                      normalized_weights[1] * target_normal_topo_feat +
                      normalized_weights[2] * target_normal_attr_feat)

        # 加权融合原型（源域原型×当前权重）
        fused_proto = (normalized_weights[0] * source_protos[0] +
                       normalized_weights[1] * source_protos[1] +
                       normalized_weights[2] * source_protos[2])

        # 融合残差（正常样本与融合原型的平均距离）
        fused_res = F.pairwise_distance(fused_feat, fused_proto.unsqueeze(0)).mean()

        # 6.4 损失函数（让融合残差逼近目标阈值，同时惩罚异常残差）
        # 主损失：MSE（残差→目标阈值）
        main_loss = F.mse_loss(fused_res, torch.tensor(target_normal_threshold, dtype=torch.float32))
        # 辅助损失：单模态残差惩罚（避免某一模态残差过大）
        text_res = F.pairwise_distance(target_normal_text_feat, source_protos[0].unsqueeze(0)).mean()
        topo_res = F.pairwise_distance(target_normal_topo_feat, source_protos[1].unsqueeze(0)).mean()
        attr_res = F.pairwise_distance(target_normal_attr_feat, source_protos[2].unsqueeze(0)).mean()
        aux_loss = (text_res + topo_res + attr_res) / 3  # 平均单模态残差
        total_loss = main_loss + 0.1 * aux_loss  # 辅助损失权重可调整

        # 6.5 反向传播+参数更新（核心：权重与编码器参数一起优化）
        total_loss.backward()
        optimizer.step()

        # 6.6 日志输出（监控权重和残差变化）
        if (epoch + 1) % 5 == 0:
            print(f"\nFinetune Epoch {epoch + 1}/{epochs}")
            print(f"Total Loss: {total_loss.item():.4f}, Main Loss: {main_loss.item():.4f}, Aux Loss: {aux_loss.item():.4f}")
            print(f"Fused Residual: {fused_res.item():.4f} (目标阈值：{target_normal_threshold})")
            print(f"可训练模态权重（归一化后）：文本={normalized_weights[0]:.3f}, 拓扑={normalized_weights[1]:.3f}, 属性={normalized_weights[2]:.3f}")
            print(f"单模态残差：文本={text_res.item():.4f}, 拓扑={topo_res.item():.4f}, 属性={attr_res.item():.4f}")

    # 步骤7：返回微调后的编码器和最终权重（去归一化前的原始权重，便于后续使用）
    final_weights = normalized_weights.detach()  # 冻结梯度，转为普通张量
    return pretrained_encoder, final_weights,source_protos


-------------------------- 异常检测推理 --------------------------
def anomaly_detection(encoder, target_dataset, modal_weights, source_protos, threshold=0.5):
    encoder.eval()
    with torch.no_grad():
        # 目标域特征编码
        target_topo_x = target_dataset.graph_data.x.float()
        target_edge_index = target_dataset.graph_data.edge_index
        target_attr_x = torch.tensor(target_dataset.attr_data, dtype=torch.float32)
        target_text_x = torch.tensor(target_dataset.text_features, dtype=torch.float32)

        text_feat, topo_feat, attr_feat = encoder(target_topo_x, target_edge_index, target_attr_x, target_text_x)
        fused_feat = modal_weights[0] * text_feat + modal_weights[1] * topo_feat + modal_weights[2] * attr_feat

        # 计算融合残差（与源域加权原型的距离）
        source_fused_proto = modal_weights[0] * source_protos[0] + modal_weights[1] * source_protos[1] + modal_weights[
            2] * source_protos[2]
        residuals = F.pairwise_distance(fused_feat, source_fused_proto.unsqueeze(0)).cpu().numpy()

        # 基于阈值生成预测标签（残差>阈值为异常）
        pred_labels = (residuals > threshold).astype(int)

        # 真实标签
        gt_labels = target_dataset.gt_labels

        # 计算指标
        precision = precision_score(gt_labels, pred_labels, zero_division=0)  # zero_division避免除以0
        recall = recall_score(gt_labels, pred_labels, zero_division=0)
        f1 = f1_score(gt_labels, pred_labels, zero_division=0)
        # AUC需要概率得分（这里用残差归一化后作为异常概率）
        anomaly_prob = (residuals - residuals.min()) / (residuals.max() - residuals.min())  # 归一化到[0,1]
        auc = roc_auc_score(gt_labels, anomaly_prob)

        # 生成指标字典
        metrics = {
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4)
        }
        anomalies = (residuals > threshold).numpy()


    return pred_labels, residuals, metrics, gt_labels, anomalies


    # 判定异常（残差>阈值为异常）

def anomaly_detection(encoder, target_dataset, modal_weights, source_protos, threshold=0.5):
    encoder.eval()
    with torch.no_grad():
        # 目标域特征编码
        target_topo_x = target_dataset.graph_data.x.float()
        target_edge_index = target_dataset.graph_data.edge_index
        target_attr_x = torch.tensor(target_dataset.attr_data, dtype=torch.float32)
        target_text_x = torch.tensor(target_dataset.text_features, dtype=torch.float32)

        text_feat, topo_feat, attr_feat = encoder(target_topo_x, target_edge_index, target_attr_x, target_text_x)
        fused_feat = modal_weights[0] * text_feat + modal_weights[1] * topo_feat + modal_weights[2] * attr_feat

        # 计算融合残差（与源域加权原型的距离）
        source_fused_proto = modal_weights[0] * source_protos[0] + modal_weights[1] * source_protos[1] + modal_weights[
            2] * source_protos[2]
        residuals = F.pairwise_distance(fused_feat, source_fused_proto.unsqueeze(0))

        # 判定异常（残差>阈值为异常）
        anomalies = (residuals > threshold).numpy()
        return anomalies, residuals.numpy()

if __name__ == "__main__":

    # 源域数据（社交网络）
    source_dataset = MultimodalGraphDataset(source_graph, source_text, source_attr, is_source_domain=True)

    # 目标域数据（金融网络）
    target_dataset = MultimodalGraphDataset(target_graph, target_text, target_attr, is_source_domain=False)

    # -------------------------- 2. 多模态因果调整对比对 --------------------------
    modal_weights, pos_pairs, neg_pairs, normal_nodes = get_modal_weights_and_pairs(source_dataset, cmi_threshold=0.05)
    print("Initial Modal Weights (from CMI):", modal_weights.numpy())
    print(f"正常节点数：{len(normal_nodes)}")
    print(f"正对比对数量：{len(pos_pairs)}")
    print(f"负对比对数量：{len(neg_pairs)}")

    # -------------------------- 3. 源域预训练 --------------------------
    encoder = MultimodalGNNEncoder(text_dim=768, topo_dim=10, attr_dim=5, hidden_dim=256, out_dim=128)
    encoder = source_pretrain(encoder, source_dataset, modal_weights, pos_pairs, neg_pairs, epochs=50)

    # -------------------------- 4. 跨域微调 --------------------------
    encoder, target_modal_weights,source_protos = cross_domain_finetune(encoder, source_dataset, target_dataset, modal_weights,
                                                          normal_nodes, epochs=20)


    pred_labels, residuals, metrics, gt_labels = anomaly_detection(encoder, target_dataset, target_modal_weights,
                                                                   source_protos, threshold=0.5)

    # 输出检测结果和指标
    print("\n" + "=" * 50)
    print("Anomaly Detection Results")
    print("=" * 50)
    print(f"预测异常数：{pred_labels.sum()}")
    print(f"真实异常数：{gt_labels.sum()}")
    print(f"Anomaly Detection Result: {pred_labels.sum()} anomalies detected")

    # 输出核心指标
    print("\n" + "=" * 50)
    print("Detection Metrics")
    print("=" * 50)
    print(f"Precision（精确率）: {metrics['Precision']:.4f}")
    print(f"Recall（召回率）: {metrics['Recall']:.4f}")
    print(f"F1-Score: {metrics['F1-Score']:.4f}")
    print(f"AUC-ROC: {metrics['AUC-ROC']:.4f}")

    # 输出详细统计
    TP = (pred_labels & gt_labels).sum()  # 真阳性（正确预测的异常）
    FP = (pred_labels & (1 - gt_labels)).sum()  # 假阳性（误判的正常）
    FN = ((1 - pred_labels) & gt_labels).sum()  # 假阴性（漏判的异常）
    TN = ((1 - pred_labels) & (1 - gt_labels)).sum()  # 真阴性（正确预测的正常）
    print("\nDetailed Statistics:")
    print(f"True Positive (TP): {TP}")
    print(f"False Positive (FP): {FP}")
    print(f"False Negative (FN): {FN}")
    print(f"True Negative (TN): {TN}")

    anomalies, residuals = anomaly_detection(encoder, target_dataset, target_modal_weights,
                                                                   source_protos, threshold=0.5)
    print('anomalies:',anomalies)
    print('residual:',residuals)
