#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2025/10/28 15:01
@File    : demo.py
@Function :
测试连接milvus和redis的脚本
！注意：
1. 启动redis的时候，是否配置了密码
2. 如果docker安装在虚拟机，需要修改下面的IP地址
"""
from pymilvus import connections, utility
import redis

# 连接到 Redis
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# 测试读写
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")
print(f"Redis value: {value}")

# 连接到 Milvus
connections.connect(host="localhost", port="19530")

# 检查版本
print(f"Milvus version: {utility.get_server_version()}")
