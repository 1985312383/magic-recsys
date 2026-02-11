"""
逻辑回归实现 - 用于CTR预估
对应教程: docs/tutorials/02_Classics/04_Logistic_Regression.md

使用6个核心统计特征：
- news_ctr: 新闻历史点击率
- user_ctr: 用户历史点击率  
- category_ctr: 类别点击率
- user_history_len: 用户历史点击数
- hour: 曝光时间（小时）
- is_new_user: 是否为新用户
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def sigmoid(z):
    """Sigmoid函数"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    """逻辑回归实现"""
    def __init__(self, learning_rate=0.01, n_epochs=100, reg_lambda=0.01):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        
    def fit(self, X, y, verbose=True):
        """训练模型（使用梯度下降）"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for epoch in range(self.n_epochs):
            # 前向传播
            z = X @ self.weights + self.bias
            y_pred = sigmoid(z)
            
            # 计算损失
            epsilon = 1e-15
            y_pred_clip = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clip) + (1 - y) * np.log(1 - y_pred_clip))
            loss += self.reg_lambda * np.sum(self.weights ** 2)
            
            # 反向传播
            dz = y_pred - y
            dw = (X.T @ dz) / n_samples + 2 * self.reg_lambda * self.weights
            db = np.mean(dz)
            
            # 更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = X @ self.weights + self.bias
        return sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


def compute_news_ctr(behaviors_df):
    """计算每篇新闻的历史点击率"""
    news_stats = defaultdict(lambda: {"impressions": 0, "clicks": 0})
    
    for _, row in behaviors_df.iterrows():
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                label = int(label)
                news_stats[news_id]["impressions"] += 1
                if label == 1:
                    news_stats[news_id]["clicks"] += 1
    
    news_ctr = {}
    for news_id, stats in news_stats.items():
        # 加1平滑，避免除零
        ctr = (stats["clicks"] + 1) / (stats["impressions"] + 10)
        news_ctr[news_id] = ctr
    
    return news_ctr


def compute_user_ctr(behaviors_df):
    """计算用户CTR"""
    user_stats = defaultdict(lambda: {"clicks": 0, "impressions": 0, "history_len": 0})
    
    for _, row in behaviors_df.iterrows():
        user_id = row["user_id"]
        
        if pd.notna(row["history"]) and row["history"].strip():
            user_stats[user_id]["history_len"] = len(row["history"].split())
        
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                _, label = item.rsplit("-", 1)
                label = int(label)
                user_stats[user_id]["impressions"] += 1
                if label == 1:
                    user_stats[user_id]["clicks"] += 1
    
    user_ctr = {}
    for user_id, stats in user_stats.items():
        if stats["impressions"] > 0:
            user_ctr[user_id] = (stats["clicks"] + 1) / (stats["impressions"] + 10)
        else:
            user_ctr[user_id] = 0.05
    
    return user_stats, user_ctr


def compute_category_ctr(behaviors_df, news_df, category2idx):
    """计算类别CTR"""
    news_category = {}
    for _, row in news_df.iterrows():
        news_category[row["news_id"]] = category2idx[row["category"]]
    
    category_stats = defaultdict(lambda: {"clicks": 0, "impressions": 0})
    
    for _, row in behaviors_df.iterrows():
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                label = int(label)
                if news_id in news_category:
                    cat_idx = news_category[news_id]
                    category_stats[cat_idx]["impressions"] += 1
                    if label == 1:
                        category_stats[cat_idx]["clicks"] += 1
    
    category_ctr = {}
    for cat_idx, stats in category_stats.items():
        category_ctr[cat_idx] = (stats["clicks"] + 1) / (stats["impressions"] + 10)
    
    return category_ctr, news_category


def extract_features_simple(behaviors_df, news_df, news_ctr, user_stats, user_ctr,
                            news_category, category_ctr, max_samples=50000):
    """
    提取特征 - 使用6个核心统计特征
    """
    features = []
    labels = []
    
    sample_count = 0
    for _, row in behaviors_df.iterrows():
        if sample_count >= max_samples:
            break
        
        user_id = row["user_id"]
        
        if pd.notna(row["impressions"]):
            for item in row["impressions"].split():
                news_id, label = item.rsplit("-", 1)
                label = int(label)
                
                if news_id not in news_category:
                    continue
                
                feat = {}
                
                # 核心特征1: 新闻历史CTR
                feat["news_ctr"] = news_ctr.get(news_id, 0.05)
                
                # 核心特征2: 用户历史CTR
                feat["user_ctr"] = user_ctr.get(user_id, 0.05)
                
                # 核心特征3: 类别CTR
                cat_idx = news_category[news_id]
                feat["category_ctr"] = category_ctr.get(cat_idx, 0.05)
                
                # 核心特征4: 用户历史长度
                feat["user_history_len"] = user_stats[user_id]["history_len"]
                
                # 核心特征5: 小时
                time_obj = pd.to_datetime(row["time"])
                feat["hour"] = time_obj.hour
                
                # 核心特征6: 是否新用户
                feat["is_new_user"] = 1 if user_stats[user_id]["history_len"] == 0 else 0
                
                features.append(feat)
                labels.append(label)
                sample_count += 1
    
    return features, labels


def vectorize_features_simple(features):
    """
    将特征列表转换为矩阵 - 6个核心特征
    """
    n_samples = len(features)
    n_features = 6
    
    X = np.zeros((n_samples, n_features))
    
    for i, feat in enumerate(features):
        X[i, 0] = feat["news_ctr"]
        X[i, 1] = feat["user_ctr"]
        X[i, 2] = feat["category_ctr"]
        X[i, 3] = min(feat["user_history_len"] / 50.0, 1.0)
        X[i, 4] = feat["hour"] / 24.0
        X[i, 5] = feat["is_new_user"]
    
    return X


def main():
    """主函数"""
    print("=" * 60)
    print("逻辑回归 CTR 预估 - 精简版（6个核心特征）")
    print("=" * 60)
    
    DATA_DIR = "../dataset/train/MINDsmall_train"
    
    print("\n[1] 加载数据...")
    behaviors = pd.read_csv(
        f"{DATA_DIR}/behaviors.tsv",
        sep="\t",
        header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"]
    )
    
    news = pd.read_csv(
        f"{DATA_DIR}/news.tsv",
        sep="\t",
        header=None,
        names=["news_id", "category", "subcategory", "title", "abstract",
               "url", "title_entities", "abstract_entities"]
    )
    
    print(f"行为日志数: {len(behaviors):,}")
    print(f"新闻数: {len(news):,}")
    
    print("\n[2] 构建类别映射...")
    categories = news["category"].unique().tolist()
    category2idx = {cat: i for i, cat in enumerate(categories)}
    print(f"类别数: {len(categories)}")
    
    print("\n[3] 计算统计特征...")
    print("  - 计算新闻CTR...")
    news_ctr = compute_news_ctr(behaviors)
    print(f"  - 已计算 {len(news_ctr):,} 篇新闻的CTR")
    
    print("  - 计算用户CTR...")
    user_stats, user_ctr = compute_user_ctr(behaviors)
    print(f"  - 已计算 {len(user_ctr):,} 个用户的CTR")
    
    print("  - 计算类别CTR...")
    category_ctr, news_category = compute_category_ctr(behaviors, news, category2idx)
    print(f"  - 已计算 {len(category_ctr)} 个类别的CTR")
    
    print("\n[4] 提取特征（6个核心特征）...")
    features, labels = extract_features_simple(
        behaviors, news, news_ctr, user_stats, user_ctr,
        news_category, category_ctr, max_samples=50000
    )
    print(f"样本数: {len(features):,}")
    print(f"正样本数: {sum(labels):,}")
    print(f"负样本数: {len(labels) - sum(labels):,}")
    print(f"CTR: {sum(labels)/len(labels):.4f}")
    
    # 显示特征示例
    print("\n特征示例（第一条）:")
    feat_example = features[0]
    print(f"  新闻CTR: {feat_example['news_ctr']:.4f}")
    print(f"  用户CTR: {feat_example['user_ctr']:.4f}")
    print(f"  类别CTR: {feat_example['category_ctr']:.4f}")
    print(f"  用户历史长度: {feat_example['user_history_len']}")
    print(f"  小时: {feat_example['hour']}")
    print(f"  是否新用户: {feat_example['is_new_user']}")
    
    print("\n[5] 特征向量化...")
    X = vectorize_features_simple(features)
    y = np.array(labels)
    print(f"特征矩阵形状: {X.shape}")
    print(f"特征名称: ['news_ctr', 'user_ctr', 'category_ctr', 'user_history_len', 'hour', 'is_new_user']")
    
    print("\n[6] 划分训练集和测试集...")
    np.random.seed(42)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"训练集: {len(y_train):,} 样本")
    print(f"测试集: {len(y_test):,} 样本")
    
    print("\n[7] 训练逻辑回归模型...")
    model = LogisticRegression(learning_rate=1.0, n_epochs=200, reg_lambda=0.01)
    model.fit(X_train, y_train)
    
    print("\n[8] 模型评估...")
    test_proba = model.predict_proba(X_test)
    test_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    auc = roc_auc_score(y_test, test_proba)
    
    print(f"测试集指标:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  AUC:       {auc:.4f}")
    
    print("\n[9] 特征重要性分析...")
    feature_names = ['news_ctr', 'user_ctr', 'category_ctr', 'user_history_len', 'hour', 'is_new_user']
    
    importance = list(zip(feature_names, model.weights))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print("特征权重（按绝对值排序）:")
    for name, weight in importance:
        print(f"  {name}: {weight:.4f}")
    
    print("\n[10] 消融实验...")
    print(f"基线 AUC（全部特征）: {auc:.4f}")
    print("\n去掉单个特征后的 AUC 变化:")

    for i, name in enumerate(feature_names):
        X_train_drop = np.delete(X_train, i, axis=1)
        X_test_drop = np.delete(X_test, i, axis=1)

        drop_model = LogisticRegression(learning_rate=1.0, n_epochs=200, reg_lambda=0.01)
        drop_model.fit(X_train_drop, y_train, verbose=False)
        drop_auc = roc_auc_score(y_test, drop_model.predict_proba(X_test_drop))

        delta = drop_auc - auc
        print(f"  去掉 {name:20s} → AUC: {drop_auc:.4f} (Δ = {delta:+.4f})")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
