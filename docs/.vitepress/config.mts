import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Magic RecSys",
  description: "A VitePress Site",
  lang: 'zh-CN',
  base: '/magic-recsys/',
  lastUpdated: true,
  head: [
    ['script', { src: 'https://hypothes.is/embed.js', async: 'true' }],
  ],
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: '主页', link: '/' },
      { text: '教程', link: '/tutorials/' }
    ],

    sidebar: [
      {
        items: [
          {
            text: '前言',
            collapsed: false,
            link: '/tutorials/'
          },
          {
            text: '第一章 核心概念（定义与标准）',
            collapsed: false,
            items: [
              { text: '推荐系统的本质与进化', link: '/tutorials/01_Concepts/01_Definition' },
              { text: '工业界核心业务场景', link: '/tutorials/01_Concepts/02_Scenarios' },
              { text: '线上与离线评测体系', link: '/tutorials/01_Concepts/03_Evaluation' },
              { text: '（选修）搜广推的关联与差异', link: '/tutorials/01_Concepts/04_Search_Ads_Basics' }
            ]
          },
          {
            text: '第二章 模型基石（数据与经典算法）',
            collapsed: true,
            items: [
              { text: '微软新闻数据集（MIND）解析', link: '/tutorials/02_Classics/01_MIND_Dataset' },
              { text: '协同过滤：纯 Numpy 实现', link: '/tutorials/02_Classics/02_CF_Numpy' },
              { text: '矩阵分解：隐向量建模', link: '/tutorials/02_Classics/03_Matrix_Factorization' },
              { text: '线性模型：CTR 预估起步', link: '/tutorials/02_Classics/04_Logistic_Regression' }
            ]
          },
          {
            text: '第三章 深度进阶（模型与检索）',
            collapsed: true,
            items: [
              { text: '特征工程：Embedding 艺术', link: '/tutorials/03_Advanced/01_Feature_Embedding' },
              { text: '手搓 DeepFM：底层算子实现', link: '/tutorials/03_Advanced/02_DeepFM_Scratch' },
              { text: 'Torch-RecHub：工业框架应用', link: '/tutorials/03_Advanced/03_Torch_RecHub_API' },
              { text: '向量召回：双塔模型与 Annoy', link: '/tutorials/03_Advanced/04_Vector_Recall_Annoy' },
              { text: '（选修）广告：多目标学习与竞价', link: '/tutorials/03_Advanced/05_Ad_Strategies_MTL' },
              { text: '（选修）搜索：Query 理解与检索', link: '/tutorials/03_Advanced/06_Search_Query_Engine' }
            ]
          },
          {
            text: '第四章 工程闭环（News-Flow 系统）',
            collapsed: true,
            items: [
              { text: '自迭代系统：架构蓝图设计', link: '/tutorials/04_Engineering/01_Flow_Architecture' },
              { text: '服务层：基于 FastAPI 的响应', link: '/tutorials/04_Engineering/02_FastAPI_Endpoint' },
              { text: '通信层：Thrift 实现解耦调用', link: '/tutorials/04_Engineering/03_Thrift_RPC' },
              { text: '存储层：Redis 在线特征查询', link: '/tutorials/04_Engineering/04_Redis_Storage' },
              { text: '闭环层：落日志与全自动迭代', link: '/tutorials/04_Engineering/05_Loop_Iteration' }
            ]
          },
          {
            text: '第五章 前沿实战（生成式推荐）',
            collapsed: true,
            items: [
              { text: 'LLM 时代的生成式推荐复现', link: '/tutorials/05_Frontiers/01_Generative_Rec_LLM' }
            ]
          },
          {
            text: '第六章 专题深潜（选修）',
            collapsed: true,
            items: [
              { text: 'A/B 测试的工业级玩法', link: '/tutorials/06_Magic_Lab/01_Advanced_AB_Test' },
              { text: '冷启动', link: '/tutorials/06_Magic_Lab/02_Cold_Start_Mastery' },
              { text: '重排与多样性', link: '/tutorials/06_Magic_Lab/03_Reranking_Strategies' },
              { text: '实时特征工程的细节', link: '/tutorials/06_Magic_Lab/04_Realtime_Features' },
              { text: '预估分数校准', link: '/tutorials/06_Magic_Lab/05_Model_Calibration' }
            ]
          }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/vuejs/vitepress' }
    ],
    search: {
      provider: 'local'
    },
    notFound: {
      title: '推荐系统崩溃了！',
      quote: '糟糕！我们的召回模型出现了严重的"精准率"问题——你要找的页面不在索引库里。别慌！让我们回到主页，用冷启动策略重新开始探索吧 🎯',
      linkLabel: '执行冷启动策略',
      linkText: '返回主页',
      code: '找不着你要找的页面咯'
    }
  },
  markdown: {
    math: true
  }
})
