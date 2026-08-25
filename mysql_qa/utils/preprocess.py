# 导入分词库
import jieba
# 导入日志
from base import logger
from typing import List


def preprocess_text(text: str) -> List[str]:
    # 预处理文本
    logger.info("开始预处理文本")
    try:
        # 分词并转换为小写
        return jieba.lcut(text.lower())
    except AttributeError as e:
        # 记录预处理失败
        logger.error(f"文本预处理失败: {e}")
        # 返回空列表
        return []


if __name__ == '__main__':
    text = "这是一个测试文本，我要吃白切鸡"
    tokens = preprocess_text(text)
    print(tokens)
