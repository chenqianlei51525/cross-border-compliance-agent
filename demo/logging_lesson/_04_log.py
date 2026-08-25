#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Vincent
@Time    : 2026/1/31 16:13
@File    : _04_log.py
@Function :
"""
import logging

# 创建日志记录器
logger = logging.getLogger("Example4")
logger.setLevel(logging.DEBUG)  # 设置记录器级别

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别

# 创建文件处理器
file_handler = logging.FileHandler('log/app.log', mode='a')
file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 为处理器设置格式
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 记录日志
logger.debug("调试信息，仅写入文件")
logger.info("程序运行正常")
logger.error("发生错误")
