#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/31 15:57
@File    : _02_log.py
@Function :
"""
import logging

# 配置日志格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 获取日志记录器
logger = logging.getLogger("Example2")

# 记录日志
logger.debug("调试模式已开启")
logger.info("正在处理数据")
logger.error("数据处理失败")
