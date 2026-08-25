#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/31 15:55
@File    : _01_log_1.py
@Function :
"""
import logging

# 配置基本的日志设置
logging.basicConfig(level=logging.WARNING)

# 获取日志记录器
logger = logging.getLogger("Example1")

# 记录不同级别的日志
logger.debug("这是调试信息，通常用于开发")
logger.info("程序运行正常")
logger.warning("注意，可能有小问题")
logger.error("发生错误")
logger.critical("严重错误，程序可能崩溃")
