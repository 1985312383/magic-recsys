"""
特征工程：Embedding 艺术

本脚本演示如何使用 Embedding 技术进行 CTR 预估：
1. 从 One-hot 编码到 Embedding 的转换
2. 使用 PyTorch 实现 Embedding 层
3. 在 MIND 数据集上训练点击预测模型
4. 可视化和分析学到的 Embedding 向量
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# 数据路径
TRAIN_DIR = "../dataset/train/MINDsmall_train"

print("=" * 60)
print("特征工程：Embedding 艺术")
print("=" * 60)

# ============================================================================
# 第一步：数据加载和准备
# ============================================================================
print("\n[1] 加载 MIND 数据集...")

# 加载新闻数据
news_df = pd.read_csv(
    f"{TRAIN_DIR}/news.tsv",
    sep="\t",
    names=["news_id", "category", "subcategory", "title", "abstract",
           "url", "title_entities", "abstract_entities"]
)

# 加载用户行为数据
behaviors_df = pd.read_csv(
    f"{TRAIN_DIR}/behaviors.tsv",
    sep="\t",
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

print(f"新闻数量: {len(news_df)}")
print(f"用户行为数量: {len(behaviors_df)}")
print(f"新闻类别: {news_df['category'].nunique()} 个")

# ============================================================================
# 第二步：构建 ID 映射
# ============================================================================
print("\n[2] 构建 ID 映射...")

# 构建用户 ID 映射
unique_users = behaviors_df['user_id'].unique()
user_to_id = {user: idx for idx, user in enumerate(unique_users)}
id_to_user = {idx: user for user, idx in user_to_id.items()}
num_users = len(user_to_id)

# 构建类别 ID 映射
unique_categories = news_df['category'].unique()
category_to_id = {cat: idx for idx, cat in enumerate(unique_categories)}
id_to_category = {idx: cat for cat, idx in category_to_id.items()}
num_categories = len(category_to_id)

print(f"用户数量: {num_users}")
print(f"类别数量: {num_categories}")
print(f"类别列表: {list(category_to_id.keys())}")

# ============================================================================
# 第三步：构建训练样本
# ============================================================================
print("\n[3] 构建训练样本...")

# 构建新闻ID到类别的映射
news_to_category = dict(zip(news_df['news_id'], news_df['category']))

# 构建训练样本
train_samples = []

for _, row in behaviors_df.iterrows():
    user_id = row['user_id']
    if user_id not in user_to_id:
        continue

    user_idx = user_to_id[user_id]
    impressions = row['impressions']

    if pd.isna(impressions):
        continue

    impressions = impressions.split()

    for imp in impressions:
        parts = imp.split('-')
        if len(parts) != 2:
            continue

        news_id, label = parts
        if news_id not in news_to_category:
            continue

        category = news_to_category[news_id]
        if category not in category_to_id:
            continue

        category_idx = category_to_id[category]
        label = int(label)

        train_samples.append((user_idx, category_idx, label))

print(f"训练样本数量: {len(train_samples)}")

# 统计正负样本比例
labels = [sample[2] for sample in train_samples]
pos_count = sum(labels)
neg_count = len(labels) - pos_count
print(f"正样本: {pos_count} ({pos_count/len(labels)*100:.2f}%)")
print(f"负样本: {neg_count} ({neg_count/len(labels)*100:.2f}%)")

# ============================================================================
# 第四步：定义 Dataset 和 DataLoader
# ============================================================================
print("\n[4] 准备数据加载器...")

class ClickDataset(Dataset):
    """点击预测数据集"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, category_idx, label = self.samples[idx]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(category_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

# 创建数据加载器
dataset = ClickDataset(train_samples)
train_loader = DataLoader(dataset, batch_size=256, shuffle=True)

print(f"Batch 数量: {len(train_loader)}")

# ============================================================================
# 第五步：定义 Embedding 模型
# ============================================================================
print("\n[5] 定义 Embedding 模型...")

class EmbeddingClickModel(nn.Module):
    """基于 Embedding 的点击预测模型"""
    def __init__(self, num_users, num_categories, embedding_dim=32):
        super().__init__()
        # 用户 Embedding 层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        # 类别 Embedding 层
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)

        # 预测层：拼接后的特征 → 点击概率
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, user_ids, category_ids):
        # 查找 Embedding
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        category_emb = self.category_embedding(category_ids)  # [batch_size, embedding_dim]

        # 拼接特征
        features = torch.cat([user_emb, category_emb], dim=1)  # [batch_size, embedding_dim*2]

        # 预测点击概率
        prob = self.fc(features)  # [batch_size, 1]
        return prob.squeeze()

# 创建模型
embedding_dim = 32
model = EmbeddingClickModel(num_users, num_categories, embedding_dim)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
print(f"用户 Embedding 参数: {num_users * embedding_dim:,}")
print(f"类别 Embedding 参数: {num_categories * embedding_dim:,}")

# ============================================================================
# 第六步：训练模型
# ============================================================================
print("\n[6] 训练模型...")

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for user_ids, category_ids, labels in train_loader:
        # 前向传播
        outputs = model(user_ids, category_ids)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

print("\n训练完成！")

# ============================================================================
# 第七步：Embedding 可视化和分析
# ============================================================================
print("\n[7] 分析学到的 Embedding...")

# 提取类别 Embedding
model.eval()
category_embeddings = model.category_embedding.weight.detach().numpy()
print(f"\n类别 Embedding 形状: {category_embeddings.shape}")

# 计算类别相似度矩阵
similarity_matrix = cosine_similarity(category_embeddings)

# 分析每个类别的最相似类别
print("\n类别相似度分析:")
print("-" * 60)
for cat, cat_idx in sorted(category_to_id.items()):
    similarities = similarity_matrix[cat_idx]
    # 排序并获取最相似的类别（排除自己）
    similar_indices = np.argsort(similarities)[::-1][1:4]  # 前3个最相似的

    print(f"\n{cat}:")
    for idx in similar_indices:
        similar_cat = id_to_category[idx]
        sim_score = similarities[idx]
        print(f"  → {similar_cat}: {sim_score:.4f}")

# 展示某个类别的 Embedding 向量
print("\n" + "=" * 60)
print("示例：sports 类别的 Embedding 向量（前10维）")
print("=" * 60)
sports_idx = category_to_id.get('sports', 0)
sports_emb = category_embeddings[sports_idx]
print(f"维度: {len(sports_emb)}")
print(f"前10维: {sports_emb[:10]}")
print(f"L2范数: {np.linalg.norm(sports_emb):.4f}")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
print("\n关键发现:")
print("1. Embedding 将高维稀疏特征（用户ID、类别ID）转换为低维稠密向量")
print("2. 训练后的 Embedding 能够自动学习语义相似性")
print("3. 相似的类别在向量空间中距离更近")
print("4. 这种表示方式为后续的深度学习模型提供了基础")

