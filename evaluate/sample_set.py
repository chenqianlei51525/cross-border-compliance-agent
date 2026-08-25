"""评测样本集（最小可跑 demo）。

正式评测需由合规业务人员基于真实法规和真实问答标注，覆盖 4 大类：
- 单认证规则查询
- 跨认证对比
- 跨文档多跳推理
- 闲聊 / 边界情况
"""

SAMPLE_QUESTIONS = [
    {
        "category": "single_cert",
        "question": "蓝牙耳机出口德国需要哪些认证？",
        "ground_truth": (
            "CE-RED（含 EN 300 328 RF + EN 301 489 EMC + EN 62368-1 LVD）、"
            "RoHS 3.0"
        ),
    },
    {
        "category": "single_cert",
        "question": "RoHS 3.0 比 2.0 多限制了哪几种物质？",
        "ground_truth": "DEHP、BBP、DBP、DIBP 四种邻苯二甲酸酯",
    },
    {
        "category": "multi_hop",
        "question": "智能音箱出口日本需要哪些认证？",
        "ground_truth": "PSE（菱形）+ 技适（TELEC 无线电）+ EMC + 安规",
    },
    {
        "category": "multi_hop",
        "question": "锂电池产品出口美国走空运需要什么文件？",
        "ground_truth": "UN38.3 测试报告 + MSDS + 1.2m 跌落测试 + IATA 货代 PI965/966/967",
    },
    {
        "category": "compliance",
        "question": "无线鼠标出口美国 FCC 怎么认证？",
        "ground_truth": "FCC Part 15 Certification, TCB 发 FCC ID, 测 RF 辐射",
    },
    {
        "category": "multi_cert",
        "question": "充电宝出口欧盟除了 CE 还需不需要 UN38.3？",
        "ground_truth": "需要,UN38.3 是运输环节强制",
    },
]
