"""
矩阵分解算法实现（FunkSVD）
"""

import numpy as np
import pandas as pd
from collections import defaultdict

# ========== FunkSVD ==========
class FunkSVD:
    """FunkSVD 矩阵分解实现"""
    
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        
    def fit(self, ratings, verbose=True):
        # 构建 ID 映射
        self.user2idx = {u: i for i, u in enumerate(ratings['user_id'].unique())}
        self.item2idx = {t: i for i, t in enumerate(ratings['item_id'].unique())}
        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: t for t, i in self.item2idx.items()}
        
        self.n_users = len(self.user2idx)
        self.n_items = len(self.item2idx)
        
        # 随机初始化隐向量
        np.random.seed(42)
        self.P = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        self.global_mean = ratings['rating'].mean()
        
        # 转换为训练数据
        train_data = [
            (self.user2idx[row['user_id']], 
             self.item2idx[row['item_id']], 
             row['rating'])
            for _, row in ratings.iterrows()
        ]
        
        # SGD 训练
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            total_loss = 0
            
            for u_idx, i_idx, rating in train_data:
                pred = self.P[u_idx] @ self.Q[i_idx]
                error = rating - pred
                
                p_u = self.P[u_idx].copy()
                q_i = self.Q[i_idx].copy()
                
                self.P[u_idx] += self.lr * (error * q_i - self.reg * p_u)
                self.Q[i_idx] += self.lr * (error * p_u - self.reg * q_i)
                
                total_loss += error ** 2 + self.reg * (np.sum(p_u**2) + np.sum(q_i**2))
            
            if verbose and (epoch + 1) % 5 == 0:
                rmse = np.sqrt(total_loss / len(train_data))
                print(f"Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")
        
        return self
    
    def predict(self, user_id, item_id):
        if user_id not in self.user2idx or item_id not in self.item2idx:
            return self.global_mean
        u_idx = self.user2idx[user_id]
        i_idx = self.item2idx[item_id]
        return self.P[u_idx] @ self.Q[i_idx]
    
    def recommend(self, user_id, n_rec=5, exclude_known=True, known_items=None):
        if user_id not in self.user2idx:
            return []
        
        u_idx = self.user2idx[user_id]
        scores = self.P[u_idx] @ self.Q.T
        
        if exclude_known and known_items:
            for item in known_items:
                if item in self.item2idx:
                    scores[self.item2idx[item]] = -np.inf
        
        top_items_idx = np.argsort(scores)[-n_rec:][::-1]
        return [self.idx2item[idx] for idx in top_items_idx]


def evaluate_mf(model, ratings_df, test_users=500, k_rec=10):
    """评估矩阵分解模型"""
    np.random.seed(42)
    
    user_items = ratings_df.groupby('user_id')['item_id'].apply(list).to_dict()
    valid_users = [u for u, items in user_items.items() if len(items) >= 2]
    test_user_ids = np.random.choice(valid_users, min(test_users, len(valid_users)), replace=False)
    
    hits = 0
    total = 0
    
    for user_id in test_user_ids:
        items = user_items[user_id]
        test_item = np.random.choice(items)
        train_items = [i for i in items if i != test_item]
        
        recs = model.recommend(user_id, n_rec=k_rec, known_items=train_items)
        
        if test_item in recs:
            hits += 1
        total += 1
    
    return hits / total


# ========== 主程序 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("矩阵分解算法实战")
    print("=" * 60)
    
    # 准备数据
    DATA_DIR = "../dataset/train/MINDsmall_train"
    
    behaviors = pd.read_csv(
        f"{DATA_DIR}/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )
    
    print("\n【1. 构建评分数据】")
    ratings_data = []
    seen = set()
    
    for idx, row in behaviors.head(20000).iterrows():
        user_id = row["user_id"]
        
        if pd.notna(row["history"]) and row["history"].strip():
            for news_id in row["history"].split():
                key = (user_id, news_id)
                if key not in seen:
                    ratings_data.append({"user_id": user_id, "item_id": news_id, "rating": 1.0})
                    seen.add(key)
        
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                if label == "1":
                    key = (user_id, news_id)
                    if key not in seen:
                        ratings_data.append({"user_id": user_id, "item_id": news_id, "rating": 1.0})
                        seen.add(key)
    
    ratings_df = pd.DataFrame(ratings_data)
    print(f"评分数据: {len(ratings_df):,} 条")
    print(f"用户数: {ratings_df['user_id'].nunique():,}")
    print(f"物品数: {ratings_df['item_id'].nunique():,}")
    
    # 训练 FunkSVD
    print("\n【2. 训练 FunkSVD】")
    model = FunkSVD(n_factors=64, n_epochs=20, lr=0.01, reg=0.01)
    model.fit(ratings_df)
    
    # 推荐示例
    print("\n【3. 推荐示例】")
    test_user = ratings_df['user_id'].iloc[0]
    user_history = ratings_df[ratings_df['user_id'] == test_user]['item_id'].tolist()
    
    print(f"用户: {test_user}")
    print(f"历史交互数: {len(user_history)}")
    print(f"历史交互（前5条）: {user_history[:5]}")
    
    recommendations = model.recommend(test_user, n_rec=5, known_items=user_history)
    print(f"FunkSVD 推荐: {recommendations}")
    
    # 评估
    print("\n【4. 评估 FunkSVD】")
    hr = evaluate_mf(model, ratings_df, test_users=500, k_rec=10)
    print(f"FunkSVD HR@10: {hr:.4f}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
