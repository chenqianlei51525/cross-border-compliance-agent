"""统一业务场景为『跨境电商合规智能问答系统』。

- VALID_SOURCES: 由"ai/java/test/ops/bigdata"改为合规业务分类
- 提供标准种子数据载入函数
"""

from .faq_seed import SEED_FAQ

VALID_SOURCES = ["ce-red", "fcc", "rohs", "un38.3", "pse", "kcc", "eup"]
DEFAULT_SOURCE = "ce-red"


def all_seed_faq():
    return SEED_FAQ


__all__ = ["VALID_SOURCES", "DEFAULT_SOURCE", "all_seed_faq"]
