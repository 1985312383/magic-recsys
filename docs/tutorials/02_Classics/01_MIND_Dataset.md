---
title: 微软新闻数据集（MIND）解析
description: 深入理解 MIND 数据集的结构与字段，为后续推荐算法实践打好数据基础。
---

# 微软新闻数据集（MIND）解析

推荐系统是"数据驱动"的技术——再精妙的算法，没有数据也是空中楼阁。从这一章开始，我们将用真实的工业级数据集来动手实践。本章选择的数据集是微软发布的 **MIND（Microsoft News Dataset）**，它是新闻推荐领域最具代表性的公开数据集之一，被学术界和工业界广泛使用。

为什么选择 MIND？首先，它规模适中、结构清晰，非常适合入门学习；其次，它包含丰富的用户行为和新闻内容信息，能支撑从协同过滤到深度学习的各类算法实验；最后，它是微软官方维护的数据集，质量有保障，且有完善的文档和社区支持。

本节将带你完成三件事：**下载数据集**、**理解每个字段的含义**、**用 Pandas 做初步的数据探索**。这是后续所有实践章节的基础，请务必跟着代码动手操作。

## 数据集背景与规模

MIND 数据集由微软新闻（MSN News）的匿名用户行为日志构建而成，时间跨度为 2019 年 10 月 12 日至 11 月 22 日，共 6 周。数据集的核心使命是作为新闻推荐的 **基准数据集（Benchmark）**，推动新闻推荐和推荐系统领域的研究进展。每篇新闻文章都包含丰富的文字内容（标题、摘要、类别、实体），每条行为日志都包含用户的点击/未点击反馈以及历史阅读记录。为了保护用户隐私，所有用户 ID 都经过安全哈希处理，与微软生产系统完全解耦。

> 📖 **学术参考**：MIND 数据集的详细说明可参考论文 [MIND: A Large-scale Dataset for News Recommendation](https://aclanthology.org/2020.acl-main.331/)（ACL 2020）。

MIND 提供两个版本：

| 版本 | 用户数 | 新闻数 | 行为日志数 | 用途 |
|------|--------|--------|------------|------|
| **MIND-large** | 100 万 | 16.1 万 | 1500 万+ | 完整实验、论文复现 |
| **MIND-small** | 5 万 | 5.1 万 | 15.7 万 | 快速上手、算法验证 |

**本教程统一使用 MIND-small**，原因很简单：它足够小（压缩包约 30MB），在普通笔记本上几秒就能加载完成，非常适合学习和调试。等你熟悉了整个流程，再切换到 MIND-large 做完整实验也很方便——两者的数据格式完全一致。

## 下载与目录结构

### 下载地址

访问 [MIND 官方页面](https://msnews.github.io/)，找到 MIND-small 的下载链接：

- 训练集：[MINDsmall_train.zip](https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip)（约 25MB）
- 验证集：[MINDsmall_dev.zip](https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip)（约 10MB）

### 目录结构

下载后解压到 `src/dataset/` 目录，结构如下：

```
src/
├── dataset/
│   ├── train/
│   │   └── MINDsmall_train/
│   │       ├── behaviors.tsv      # 用户行为日志（核心）
│   │       ├── news.tsv           # 新闻内容信息（核心）
│   │       ├── entity_embedding.vec   # 实体向量（选修）
│   │       └── relation_embedding.vec # 关系向量（选修）
│   └── dev/
│       └── MINDsmall_dev/
│           └── ...（同上）
└── ch02/
    └── 01_mind_exploration.py     # 本节代码
```

本节重点讲解 `behaviors.tsv` 和 `news.tsv`，这两个文件包含了推荐系统最核心的信息：**谁在什么时候看到了哪些新闻，以及这些新闻长什么样**。

## 字段详解：behaviors.tsv

`behaviors.tsv` 是用户行为日志，记录了"谁在什么时候看到了哪些新闻，点了哪些、没点哪些"。这是推荐系统最核心的数据——没有用户行为，就没有个性化推荐。

先用 Pandas 加载数据：

```python
import pandas as pd

# 数据路径（相对于 src/ch02/ 目录）
TRAIN_DIR = "../dataset/train/MINDsmall_train"

# 加载行为日志
behaviors = pd.read_csv(
    f"{TRAIN_DIR}/behaviors.tsv",
    sep="\t",
    header=None,
    names=["impression_id", "user_id", "time", "history", "impressions"]
)

print(f"行为日志总数: {len(behaviors):,}")
```

```
行为日志总数: 156,965
```

每一行代表一次"曝光事件"（Impression），包含 5 个字段：

| 字段 | 含义 | 示例 |
|------|------|------|
| `impression_id` | 曝光事件的唯一标识 | `1` |
| `user_id` | 用户的匿名 ID | `U13740` |
| `time` | 曝光发生的时间 | `11/11/2019 9:05:58 AM` |
| `history` | 用户的历史点击序列（空格分隔的新闻 ID） | `N55189 N42782 N34694...` |
| `impressions` | 本次曝光的新闻列表及点击情况 | `N55689-1 N35729-0` |

重点解释 `impressions` 字段，它是推荐系统建模的核心：

- 格式：`新闻ID-标签`，多个新闻用空格分隔
- 标签含义：`1` 表示用户点击了，`0` 表示曝光了但用户没点
- 示例：`N55689-1 N35729-0` 表示 N55689 被点击，N35729 未被点击

这里的 **`-1` 就是正样本，`-0` 就是负样本**——这是监督学习最基本的训练数据格式。推荐系统的核心任务，就是学习一个模型，能够区分"用户会点击的新闻"和"用户不会点击的新闻"。

### 正负样本统计

让我们解析 impressions 字段，统计正负样本的分布：

```python
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
```

```
正样本（点击）总数: 236,344
负样本（未点击）总数: 5,607,100
正负样本比例: 1:23.7
```

**关键洞察**：正负样本比例约为 **1:24**，这在推荐场景中是非常典型的——大多数曝光的内容用户并不会点击。这种"样本不均衡"问题，是推荐系统建模时需要特别处理的。

## 字段详解：news.tsv

`news.tsv` 是新闻内容信息，描述了每一篇新闻"长什么样"——标题、类别、摘要等。这些信息在推荐系统中被称为"物品特征"（Item Features）。

```python
# 加载新闻数据
news = pd.read_csv(
    f"{TRAIN_DIR}/news.tsv",
    sep="\t",
    header=None,
    names=["news_id", "category", "subcategory", "title", "abstract", 
           "url", "title_entities", "abstract_entities"]
)

print(f"新闻总数: {len(news):,}")
```

```
新闻总数: 51,282
```

每一行代表一篇新闻，包含 8 个字段：

| 字段 | 含义 | 用途 |
|------|------|------|
| `news_id` | 新闻的唯一标识 | 关联行为日志 |
| `category` | 一级类别 | 粗粒度内容分类 |
| `subcategory` | 二级类别 | 细粒度内容分类 |
| `title` | 新闻标题 | NLP 特征提取 |
| `abstract` | 新闻摘要 | NLP 特征提取 |
| `url` | 原文链接 | 参考（部分已失效） |
| `title_entities` | 标题中的实体（JSON 格式） | 知识图谱增强 |
| `abstract_entities` | 摘要中的实体（JSON 格式） | 知识图谱增强 |

样例数据：

```python
print(news[["news_id", "category", "subcategory", "title"]].head(3))
```

```
  news_id   category      subcategory                                              title
0  N55528  lifestyle  lifestyleroyals  The Brands Queen Elizabeth, Prince Charles, ...
1  N19639     health       weightloss                    50 Worst Habits For Belly Fat
2  N61837       news        newsworld  The Cost of Trump's Aid Freeze in the Trench...
```

### 实体字段详解（选修）

`title_entities` 和 `abstract_entities` 是从 WikiData 知识图谱中抽取的实体信息，格式为 JSON 数组。这是 MIND 数据集的一大亮点——它不仅提供了文本内容，还提供了结构化的知识信息，方便做知识增强的推荐研究。

一个实体的 JSON 结构如下：

```json
{
  "Label": "PGA Tour",           // WikiData 中的实体名称
  "Type": "O",                   // 实体类型
  "WikidataId": "Q910409",       // WikiData 实体 ID
  "Confidence": 1.0,             // 实体链接的置信度
  "OccurrenceOffsets": [0],      // 实体在文本中的字符位置偏移
  "SurfaceForms": ["PGA Tour"]   // 原文中的实体表述形式
}
```

对于入门学习，我们暂时用不到这两个字段。但如果你想做知识增强的推荐研究，这些信息会非常有价值。

### 嵌入文件详解（选修）

除了 `behaviors.tsv` 和 `news.tsv`，数据集还提供了两个预训练的嵌入文件：

- **`entity_embedding.vec`**：实体的 100 维向量表示
- **`relation_embedding.vec`**：实体间关系的 100 维向量表示

这些向量是用 **TransE** 方法从 WikiData 子图中学习得到的。文件格式为：第一列是实体/关系 ID，后续 100 列是嵌入向量值。例如：`Q42306013  0.014516 -0.106958 0.024590 ... -0.080382`。

> ⚠️ **注意**：由于子图学习的原因，部分实体可能在 `entity_embedding.vec` 中没有对应的嵌入向量。

## 动手实践：数据探索

理解了字段含义，接下来用 Pandas 做一些基础的数据探索，为后续建模做准备。

### 任务一：统计新闻类别分布

新闻类别是最直观的内容特征，了解类别分布有助于理解数据集的整体构成。

```python
# 统计一级类别分布
category_counts = news["category"].value_counts()
print("新闻类别分布（Top 15）:")
print(category_counts.head(15))
```

```
新闻类别分布（Top 15）:
news             15774
sports           14510
finance           3107
foodanddrink      2551
lifestyle         2479
travel            2350
video             2068
weather           2048
health            1885
autos             1639
tv                 889
music              769
movies             606
entertainment      587
kids                17
```

**关键洞察**：`news`（时事新闻）类别占比最高，约 31%；其次是 `sports`（体育）约 28%。这两个类别占据了近 60% 的内容，说明 MSN News 的用户主要关注时事和体育。

### 任务二：分析用户活跃度

用户活跃度决定了我们有多少行为数据可用于建模。活跃用户的历史行为丰富，模型更容易学到他们的兴趣；而低活跃用户的数据稀疏，是推荐系统的难点（冷启动问题）。

```python
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
```

```
用户总数: 50,000
历史点击数 - 均值: 18.5
历史点击数 - 中位数: 11.0
历史点击数 - 最大值: 558
```

```python
# 分段统计
bins = [0, 5, 10, 20, 50, 100, float("inf")]
labels = ["1-5", "6-10", "11-20", "21-50", "51-100", ">100"]
activity_dist = pd.cut(user_activity, bins=bins, labels=labels).value_counts().sort_index()
print("用户活跃度分布（按历史点击数分段）:")
print(activity_dist)
```

```
用户活跃度分布（按历史点击数分段）:
1-5       12087
6-10      11823
11-20     11498
21-50     10064
51-100     2871
>100        765
```

**关键洞察**：用户活跃度呈现典型的**长尾分布**——约 48% 的用户历史点击在 10 条以内（1-5 + 6-10），而只有 7% 的用户点击超过 50 条。这种分布特点决定了推荐系统需要同时处理"数据丰富"和"数据稀疏"两种情况。

### 任务三：构建用户-新闻交互数据

为后续的协同过滤算法做准备，我们需要构建一个"用户-新闻交互数据"，记录每个用户点击过哪些新闻。

```python
def build_interaction_data(behaviors_df):
    """从行为日志中提取用户-新闻交互对"""
    interactions = []
    
    for idx, row in behaviors_df.iterrows():
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

# 为加快演示速度，只处理前 1 万条行为
interactions = build_interaction_data(behaviors.head(10000))
print(f"交互记录总数: {len(interactions):,}")
print(f"其中正样本数: {(interactions['label']==1).sum():,}")
print(f"其中负样本数: {(interactions['label']==0).sum():,}")
```

```
交互记录总数: 691,197
其中正样本数: 337,718
其中负样本数: 353,479
```

```python
print("交互数据样例:")
print(interactions.head(10))
```

```
交互数据样例:
  user_id news_id  label
0  U13740  N55189      1
1  U13740  N42782      1
2  U13740  N34694      1
3  U13740  N45794      1
4  U13740  N18445      1
5  U13740  N63302      1
6  U13740  N10414      1
7  U13740  N19347      1
8  U13740  N31801      1
9  U13740  N55689      1
```

这个 `interactions` DataFrame 就是后续协同过滤、矩阵分解算法的输入数据。每一行记录了"用户 X 对新闻 Y 的反馈是 Z（1=点击，0=未点击）"。

## 数据集的局限性

MIND 是优秀的学习数据集，但它也有一些局限性，在实际使用时需要注意：

1. **只有英文新闻**：MIND 来源于 MSN News 的英文版，不包含中文内容。

2. **只有点击信号**：MIND 只记录了"点击/未点击"，没有更丰富的反馈信号（如阅读时长、评论、分享）。

3. **无新闻正文**：由于 MSN 新闻的版权限制，数据集**不提供新闻全文**，只有标题和摘要。微软官方提供了一个辅助脚本用于从 URL 解析网页内容，但由于时间久远，大部分 URL 已失效。

4. **时间跨度有限**：数据集只覆盖 6 周，无法研究长期用户兴趣演变。

尽管如此，MIND 仍然是新闻推荐领域最标准的 benchmark 数据集，后续章节的所有算法都将基于它展开。

---

**下一节预告**：有了数据，就可以开始写算法了。下一节我们将用纯 Numpy 实现经典的协同过滤算法，不依赖任何机器学习框架，从零理解"找相似用户、推荐他们喜欢的内容"的核心逻辑。
