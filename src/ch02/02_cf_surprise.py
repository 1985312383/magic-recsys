"""
协同过滤算法实现（Surprise 框架版）
使用 Surprise 库快速实现 UserCF 和 ItemCF
"""

import pandas as pd
from collections import defaultdict
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# ========== 数据准备 ==========
print("=" * 60)
print("协同过滤算法实战（Surprise 框架）")
print("=" * 60)

DATA_DIR = "../dataset/train/MINDsmall_train"

behaviors = pd.read_csv(
    f"{DATA_DIR}/behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

# 构建评分数据（点击=1）
print("\n【1. 构建评分数据】")
ratings_data = []
seen = set()

for idx, row in behaviors.head(20000).iterrows():
    user_id = row["user_id"]
    
    if pd.notna(row["history"]) and row["history"].strip():
        for news_id in row["history"].split():
            key = (user_id, news_id)
            if key not in seen:
                ratings_data.append((user_id, news_id, 1.0))
                seen.add(key)
    
    if pd.notna(row["impressions"]):
        for item in row["impressions"].split():
            news_id, label = item.rsplit("-", 1)
            if label == "1":
                key = (user_id, news_id)
                if key not in seen:
                    ratings_data.append((user_id, news_id, 1.0))
                    seen.add(key)

ratings_df = pd.DataFrame(ratings_data, columns=["user_id", "item_id", "rating"])
print(f"评分数据条数: {len(ratings_df):,}")
print(f"用户数: {ratings_df['user_id'].nunique():,}")
print(f"物品数: {ratings_df['item_id'].nunique():,}")

# 转换为 Surprise 数据集
reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(ratings_df, reader)

# 划分训练集和测试集
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
print(f"\n训练集大小: {trainset.n_ratings:,}")
print(f"测试集大小: {len(testset):,}")

# ========== UserCF ==========
print("\n" + "=" * 60)
print("【2. UserCF（Surprise 实现）】")
print("=" * 60)

# 配置 UserCF
sim_options_user = {
    "name": "cosine",      # 相似度计算方式
    "user_based": True,    # 基于用户
    "min_support": 1       # 最少共同评分物品数
}

user_cf = KNNBasic(k=20, sim_options=sim_options_user, verbose=False)

print("\n训练 UserCF 模型...")
user_cf.fit(trainset)
print("训练完成！")

# 预测和评估
print("\n在测试集上评估...")
predictions_user = user_cf.test(testset)
rmse_user = accuracy.rmse(predictions_user, verbose=False)
mae_user = accuracy.mae(predictions_user, verbose=False)
print(f"  RMSE: {rmse_user:.4f}")
print(f"  MAE: {mae_user:.4f}")

# 为某个用户推荐
def get_top_n(predictions, n=5):
    """从预测结果中获取每个用户的 Top-N 推荐"""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# 获取推荐结果示例
print("\n--- UserCF 推荐示例 ---")
# 使用全量数据训练后推荐
full_trainset = data.build_full_trainset()
user_cf_full = KNNBasic(k=20, sim_options=sim_options_user, verbose=False)
user_cf_full.fit(full_trainset)

# 获取所有用户未评分的物品并预测
testset_full = full_trainset.build_anti_testset()
predictions_full = user_cf_full.test(testset_full)
top_n_user = get_top_n(predictions_full, n=5)

# 展示一个用户的推荐
sample_user = list(top_n_user.keys())[0]
print(f"用户 {sample_user} 的 UserCF 推荐:")
for rank, (item, score) in enumerate(top_n_user[sample_user], 1):
    print(f"  {rank}. {item} (预测分: {score:.4f})")

# ========== ItemCF ==========
print("\n" + "=" * 60)
print("【3. ItemCF（Surprise 实现）】")
print("=" * 60)

sim_options_item = {
    "name": "cosine",
    "user_based": False,   # 基于物品
    "min_support": 1
}

item_cf = KNNBasic(k=20, sim_options=sim_options_item, verbose=False)

print("\n训练 ItemCF 模型...")
item_cf.fit(trainset)
print("训练完成！")

print("\n在测试集上评估...")
predictions_item = item_cf.test(testset)
rmse_item = accuracy.rmse(predictions_item, verbose=False)
mae_item = accuracy.mae(predictions_item, verbose=False)
print(f"  RMSE: {rmse_item:.4f}")
print(f"  MAE: {mae_item:.4f}")

# ItemCF 推荐示例
print("\n--- ItemCF 推荐示例 ---")
item_cf_full = KNNBasic(k=20, sim_options=sim_options_item, verbose=False)
item_cf_full.fit(full_trainset)
predictions_full_item = item_cf_full.test(testset_full)
top_n_item = get_top_n(predictions_full_item, n=5)

print(f"用户 {sample_user} 的 ItemCF 推荐:")
for rank, (item, score) in enumerate(top_n_item[sample_user], 1):
    print(f"  {rank}. {item} (预测分: {score:.4f})")

# ========== 对比总结 ==========
print("\n" + "=" * 60)
print("【4. 算法对比】")
print("=" * 60)

print(f"""
| 算法    | RMSE   | MAE    |
|---------|--------|--------|
| UserCF  | {rmse_user:.4f} | {mae_user:.4f} |
| ItemCF  | {rmse_item:.4f} | {mae_item:.4f} |
""")

print("=" * 60)
print("Surprise 实现完成！")
print("=" * 60)
