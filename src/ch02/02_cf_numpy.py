"""
协同过滤算法实现（Numpy 版）
包含 UserCF 和 ItemCF 的完整实现
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import time

# ========== 数据加载 ==========
print("=" * 60)
print("协同过滤算法实战（Numpy 实现）")
print("=" * 60)

DATA_DIR = "../dataset/train/MINDsmall_train"

# 加载行为数据
behaviors = pd.read_csv(
    f"{DATA_DIR}/behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

# ========== 构建交互矩阵 ==========
print("\n【1. 构建用户-物品交互数据】")

def build_interactions(behaviors_df, max_rows=20000):
    """从行为日志构建交互数据（只用正样本）"""
    user_items = defaultdict(set)
    
    for idx, row in behaviors_df.head(max_rows).iterrows():
        user_id = row["user_id"]
        
        # 历史点击
        if pd.notna(row["history"]) and row["history"].strip():
            for news_id in row["history"].split():
                user_items[user_id].add(news_id)
        
        # 当前曝光中的点击
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                if label == "1":
                    user_items[user_id].add(news_id)
    
    return user_items

user_items = build_interactions(behaviors)

# 构建 ID 映射
all_users = list(user_items.keys())
all_items = set()
for items in user_items.values():
    all_items.update(items)
all_items = list(all_items)

user2idx = {u: i for i, u in enumerate(all_users)}
idx2user = {i: u for u, i in user2idx.items()}
item2idx = {t: i for i, t in enumerate(all_items)}
idx2item = {i: t for t, i in item2idx.items()}

n_users = len(all_users)
n_items = len(all_items)

print(f"用户数: {n_users:,}")
print(f"物品数: {n_items:,}")
print(f"交互数: {sum(len(v) for v in user_items.values()):,}")

# 构建稀疏交互矩阵（用户 x 物品）
print("\n构建交互矩阵...")
interaction_matrix = np.zeros((n_users, n_items), dtype=np.float32)
for user, items in user_items.items():
    u_idx = user2idx[user]
    for item in items:
        i_idx = item2idx[item]
        interaction_matrix[u_idx, i_idx] = 1.0

print(f"交互矩阵形状: {interaction_matrix.shape}")
print(f"稀疏度: {1 - interaction_matrix.sum() / (n_users * n_items):.4%}")

# ========== UserCF 实现 ==========
print("\n" + "=" * 60)
print("【2. UserCF：基于用户的协同过滤】")
print("=" * 60)

def cosine_similarity_matrix(matrix):
    """计算余弦相似度矩阵"""
    # 归一化：每行除以其 L2 范数
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized = matrix / norms
    # 相似度 = 归一化矩阵 × 归一化矩阵的转置
    similarity = np.dot(normalized, normalized.T)
    return similarity

print("\n计算用户相似度矩阵...")
start_time = time.time()
user_similarity = cosine_similarity_matrix(interaction_matrix)
print(f"计算耗时: {time.time() - start_time:.2f}s")
print(f"用户相似度矩阵形状: {user_similarity.shape}")

# 对角线置零（自己和自己的相似度不参与推荐）
np.fill_diagonal(user_similarity, 0)

def user_cf_recommend(user_idx, interaction_matrix, user_similarity, k=10, n_rec=5):
    """
    UserCF 推荐
    Args:
        user_idx: 目标用户索引
        interaction_matrix: 用户-物品交互矩阵
        user_similarity: 用户相似度矩阵
        k: 使用 top-k 相似用户
        n_rec: 推荐物品数量
    Returns:
        推荐物品索引列表
    """
    # 找到 top-k 相似用户
    sim_scores = user_similarity[user_idx]
    top_k_users = np.argsort(sim_scores)[-k:][::-1]
    
    # 目标用户已交互的物品
    user_interacted = set(np.where(interaction_matrix[user_idx] > 0)[0])
    
    # 聚合相似用户的物品偏好
    item_scores = defaultdict(float)
    for sim_user in top_k_users:
        sim = sim_scores[sim_user]
        if sim <= 0:
            continue
        sim_user_items = np.where(interaction_matrix[sim_user] > 0)[0]
        for item_idx in sim_user_items:
            if item_idx not in user_interacted:
                item_scores[item_idx] += sim
    
    # 按分数排序，返回 top-n
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items[:n_rec]]

# 测试 UserCF
print("\n--- UserCF 推荐示例 ---")
test_user_idx = 0
test_user_id = idx2user[test_user_idx]
user_history = [idx2item[i] for i in np.where(interaction_matrix[test_user_idx] > 0)[0][:5]]

print(f"目标用户: {test_user_id}")
print(f"历史点击（前5条）: {user_history}")

# 找相似用户
sim_scores = user_similarity[test_user_idx]
top_similar_users = np.argsort(sim_scores)[-3:][::-1]
print(f"最相似的3个用户:")
for rank, sim_user_idx in enumerate(top_similar_users, 1):
    print(f"  {rank}. {idx2user[sim_user_idx]} (相似度: {sim_scores[sim_user_idx]:.4f})")

# 推荐
recommendations = user_cf_recommend(test_user_idx, interaction_matrix, user_similarity, k=20, n_rec=5)
print(f"UserCF 推荐结果:")
for rank, item_idx in enumerate(recommendations, 1):
    print(f"  {rank}. {idx2item[item_idx]}")

# ========== ItemCF 实现 ==========
print("\n" + "=" * 60)
print("【3. ItemCF：基于物品的协同过滤】")
print("=" * 60)

print("\n计算物品相似度矩阵...")
start_time = time.time()
# 物品相似度 = 交互矩阵转置后计算
item_similarity = cosine_similarity_matrix(interaction_matrix.T)
print(f"计算耗时: {time.time() - start_time:.2f}s")
print(f"物品相似度矩阵形状: {item_similarity.shape}")

np.fill_diagonal(item_similarity, 0)

def item_cf_recommend(user_idx, interaction_matrix, item_similarity, n_rec=5):
    """
    ItemCF 推荐
    Args:
        user_idx: 目标用户索引
        interaction_matrix: 用户-物品交互矩阵
        item_similarity: 物品相似度矩阵
        n_rec: 推荐物品数量
    Returns:
        推荐物品索引列表
    """
    # 用户已交互的物品
    user_interacted = np.where(interaction_matrix[user_idx] > 0)[0]
    user_interacted_set = set(user_interacted)
    
    # 聚合已交互物品的相似物品
    item_scores = defaultdict(float)
    for item_idx in user_interacted:
        sim_scores = item_similarity[item_idx]
        # 找相似物品
        similar_items = np.argsort(sim_scores)[-50:][::-1]
        for sim_item in similar_items:
            if sim_item not in user_interacted_set:
                item_scores[sim_item] += sim_scores[sim_item]
    
    # 按分数排序
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items[:n_rec]]

# 测试 ItemCF
print("\n--- ItemCF 推荐示例 ---")
print(f"目标用户: {test_user_id}")
print(f"历史点击（前5条）: {user_history}")

# 展示一个物品的相似物品
sample_item_idx = np.where(interaction_matrix[test_user_idx] > 0)[0][0]
sample_item_id = idx2item[sample_item_idx]
sim_scores = item_similarity[sample_item_idx]
top_similar_items = np.argsort(sim_scores)[-3:][::-1]
print(f"\n物品 {sample_item_id} 的相似物品:")
for rank, sim_item_idx in enumerate(top_similar_items, 1):
    print(f"  {rank}. {idx2item[sim_item_idx]} (相似度: {sim_scores[sim_item_idx]:.4f})")

# 推荐
recommendations = item_cf_recommend(test_user_idx, interaction_matrix, item_similarity, n_rec=5)
print(f"\nItemCF 推荐结果:")
for rank, item_idx in enumerate(recommendations, 1):
    print(f"  {rank}. {idx2item[item_idx]}")

# ========== 评估指标 ==========
print("\n" + "=" * 60)
print("【4. 离线评估】")
print("=" * 60)

def evaluate_cf(recommend_func, interaction_matrix, similarity_matrix, 
                test_users=100, k_rec=10, is_user_cf=True):
    """
    评估协同过滤算法
    使用留一法：对每个用户，隐藏一个正样本作为测试
    """
    hits = 0
    total = 0
    precision_sum = 0
    recall_sum = 0
    
    np.random.seed(42)
    test_user_indices = np.random.choice(n_users, min(test_users, n_users), replace=False)
    
    for user_idx in test_user_indices:
        user_items_idx = np.where(interaction_matrix[user_idx] > 0)[0]
        if len(user_items_idx) < 2:
            continue
        
        # 随机选一个作为测试集
        test_item = np.random.choice(user_items_idx)
        
        # 临时移除测试物品
        original_value = interaction_matrix[user_idx, test_item]
        interaction_matrix[user_idx, test_item] = 0
        
        # 获取推荐
        if is_user_cf:
            recs = recommend_func(user_idx, interaction_matrix, similarity_matrix, k=20, n_rec=k_rec)
        else:
            recs = recommend_func(user_idx, interaction_matrix, similarity_matrix, n_rec=k_rec)
        
        # 恢复
        interaction_matrix[user_idx, test_item] = original_value
        
        # 计算指标
        if test_item in recs:
            hits += 1
        
        total += 1
        precision_sum += (1 if test_item in recs else 0) / k_rec
        recall_sum += (1 if test_item in recs else 0) / 1  # 测试集只有1个
    
    hit_rate = hits / total if total > 0 else 0
    precision = precision_sum / total if total > 0 else 0
    recall = recall_sum / total if total > 0 else 0
    
    return {
        "HR@{}".format(k_rec): hit_rate,
        "Precision@{}".format(k_rec): precision,
        "Recall@{}".format(k_rec): recall,
        "测试用户数": total
    }

print("\n评估 UserCF...")
user_cf_metrics = evaluate_cf(user_cf_recommend, interaction_matrix, user_similarity,
                               test_users=500, k_rec=10, is_user_cf=True)
for k, v in user_cf_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n评估 ItemCF...")
item_cf_metrics = evaluate_cf(item_cf_recommend, interaction_matrix, item_similarity,
                               test_users=500, k_rec=10, is_user_cf=False)
for k, v in item_cf_metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("Numpy 实现完成！")
print("=" * 60)
