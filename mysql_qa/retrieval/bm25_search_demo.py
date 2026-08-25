#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/8 18:29
@File    : bm25_search_demo.py
@Function :
"""
import jieba
from rank_bm25 import BM25L
from typing import List
from base import logger


class BM25SearchDemo:
    def __init__(self, documents: List[str]):
        # 初始化文档集合
        self.documents = documents
        # 分词后的文档
        self.tokenized_docs = [jieba.lcut(doc) for doc in documents]  # List[List[str]]
        # 初始化BM25模型
        self.bm25 = BM25L(self.tokenized_docs)
        logger.info("BM25模型初始化完成")

    def search(self, query):
        # 分词查询
        tokenized_query = jieba.lcut(query)  # List[str]
        try:
            # 计算每个文档的BM25得分
            scores = self.bm25.get_scores(tokenized_query)
            print(f'scores--》{scores}')
            # 获取最高得分的文档索引
            best_idx = scores.argmax()
            best_score = scores[best_idx]
            best_doc = self.documents[best_idx]
            logger.info(f"查询: {query}, 最佳匹配: {best_doc}, 得分: {best_score}")
            return best_doc, best_score
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return None, 0.0


def main():
    # 示例文档集合
    documents = ["我喜欢编程", "编程很有趣", "今天天气很好"]
    # 初始化BM25检索器
    bm25_search = BM25SearchDemo(documents)
    # 示例查询
    query = "他喜欢编程"
    # 执行检索
    result, score = bm25_search.search(query)
    if result:
        logger.info(f"查询结果: {result}, 得分: {score}")
    else:
        logger.info("未找到匹配结果")


if __name__ == "__main__":
    main()
