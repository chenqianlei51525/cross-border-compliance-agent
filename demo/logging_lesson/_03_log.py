#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/31 15:59
@File    : _03_log.py
@Function :
"""
import logging

# 配置日志，输出到文件
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # 日志文件路径
    filemode='a',  # 'a'表示追加，'w'表示覆盖
    encoding='utf-8'
)

# 获取日志记录器
logger = logging.getLogger("Example3")

# 记录日志
logger.info("程序启动")
logger.warning("内存使用率较高")
logger.error("无法连接数据库")
