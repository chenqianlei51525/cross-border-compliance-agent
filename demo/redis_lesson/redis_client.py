#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/31 18:09
@File    : redis_client.py
@Function :
"""
import redis
import json
from base import Config, logger


class RedisClient:
    def __init__(self):
        self.logger = logger
        try:
            self.client = redis.StrictRedis(
                host=Config().REDIS_HOST,
                port=Config().REDIS_PORT,
                password=Config().REDIS_PASSWORD,
                db=Config().REDIS_DB,
                decode_responses=True
            )
            self.logger.info("Redis 连接成功")
        except redis.RedisError as e:
            self.logger.error(f"Redis 连接失败: {e}")
            raise

    def set_data(self, key, value):
        try:
            self.client.set(key, json.dumps(value))
            self.logger.info(f"存储数据到 Redis: {key}")
        except redis.RedisError as e:
            self.logger.error(f"Redis 存储失败: {e}")

    def get_data(self, key):
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except redis.RedisError as e:
            self.logger.error(f"Redis 获取失败: {e}")
            return None

    def get_answer(self, query):
        try:
            answer = self.client.get(f"answer:{query}")
            if answer:
                self.logger.info(f"从 Redis 获取答案: {query}")
                return answer
            return None
        except redis.RedisError as e:
            self.logger.error(f"Redis 查询失败: {e}")
            return None


import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # 初始化 Redis 客户端
    redis_client = RedisClient()
    # 示例数据
    key = "username"
    value = {"name": "兔子行", "age": 25}
    # 存储数据
    redis_client.set_data(key, value)
    # 获取数据
    result = redis_client.get_data(key)
    if result:
        logger.info(f"查询结果: {result}")
    else:
        logger.info("未找到数据")
    # 示例查询缓存
    query = "test_query"
    answer = redis_client.get_answer(query)
    if answer:
        logger.info(f"缓存答案: {answer}")
    else:
        logger.info("未找到缓存答案")


if __name__ == "__main__":
    main()
