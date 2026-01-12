"""
MIND 数据集探索
本脚本用于探索 MIND-small 数据集的基本结构和统计信息
"""

import pandas as pd
from collections import defaultdict

# 数据路径
DATA_DIR = "../dataset"
TRAIN_DIR = f"{DATA_DIR}/train/MINDsmall_train"
DEV_DIR = f"{DATA_DIR}/dev/MINDsmall_dev"

print("=" * 60)
print("MIND-small 数据集探索")
print("=" * 60)

# ========== 1. 加载数据 ==========
print("\n【1. 加载数据】")

# 加载行为日志
behaviors = pd.read_csv(
    f"{TRAIN_DIR}/behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)
print(f"行为日志总数: {len(behaviors):,}")

# 加载新闻数据
news = pd.read_csv(
    f"{TRAIN_DIR}/news.tsv",
    sep="\t",
    header=None,
    names=["news_id", "category", "subcategory", "title", "abstract", 
           "url", "title_entities", "abstract_entities"]
)
print(f"新闻总数: {len(news):,}")

# ========== 2. behaviors.tsv 字段解析 ==========
print("\n【2. behaviors.tsv 样例】")
print(behaviors.head(3).to_string())

# ========== 3. 正负样本统计 ==========
print("\n【3. 正负样本统计】")

def parse_impressions(impressions_str):
    """解析 impressions 字段，返回正样本数和负样本数"""
    if pd.isna(impressions_str):
        return 0, 0
    items = impressions_str.split()
    pos_count = sum(1 for item in items if item.endswith("-1"))
    neg_count = sum(1 for item in items if item.endswith("-0"))
    return pos_count, neg_count

behaviors[["pos_count", "neg_count"]] = behaviors["impressions"].apply(
    lambda x: pd.Series(parse_impressions(x))
)

total_pos = behaviors["pos_count"].sum()
total_neg = behaviors["neg_count"].sum()

print(f"正样本（点击）总数: {total_pos:,}")
print(f"负样本（未点击）总数: {total_neg:,}")
print(f"正负样本比例: 1:{total_neg/total_pos:.1f}")

# ========== 4. news.tsv 字段解析 ==========
print("\n【4. news.tsv 样例】")
# 只显示关键字段
print(news[["news_id", "category", "subcategory", "title"]].head(3).to_string())

# ========== 5. 新闻类别分布 ==========
print("\n【5. 新闻类别分布（Top 15）】")
category_counts = news["category"].value_counts()
print(category_counts.head(15).to_string())

# ========== 6. 用户活跃度分析 ==========
print("\n【6. 用户活跃度分析】")

def count_history(history_str):
    if pd.isna(history_str) or history_str.strip() == "":
        return 0
    return len(history_str.split())

behaviors["history_len"] = behaviors["history"].apply(count_history)
user_activity = behaviors.groupby("user_id")["history_len"].max()

print(f"用户总数: {len(user_activity):,}")
print(f"历史点击数 - 均值: {user_activity.mean():.1f}")
print(f"历史点击数 - 中位数: {user_activity.median():.1f}")
print(f"历史点击数 - 最大值: {user_activity.max()}")

# 分段统计
bins = [0, 5, 10, 20, 50, 100, float("inf")]
labels = ["1-5", "6-10", "11-20", "21-50", "51-100", ">100"]
activity_dist = pd.cut(user_activity, bins=bins, labels=labels).value_counts().sort_index()
print("\n用户活跃度分布（按历史点击数分段）:")
print(activity_dist.to_string())

# ========== 7. 构建交互数据 ==========
print("\n【7. 构建用户-新闻交互数据】")

def build_interaction_data(behaviors_df, sample_size=10000):
    """从行为日志中提取用户-新闻交互对（采样以加速）"""
    interactions = []
    
    for idx, row in behaviors_df.head(sample_size).iterrows():
        user_id = row["user_id"]
        
        # 从历史点击中提取
        if pd.notna(row["history"]) and row["history"].strip():
            for news_id in row["history"].split():
                interactions.append((user_id, news_id, 1))
        
        # 从当前曝光中提取
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                interactions.append((user_id, news_id, int(label)))
    
    return pd.DataFrame(interactions, columns=["user_id", "news_id", "label"])

interactions = build_interaction_data(behaviors)
print(f"交互记录总数（前1万条行为）: {len(interactions):,}")
print(f"其中正样本数: {(interactions['label']==1).sum():,}")
print(f"其中负样本数: {(interactions['label']==0).sum():,}")

print("\n交互数据样例:")
print(interactions.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("数据探索完成！")
print("=" * 60)
